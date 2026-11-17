"""Explicit small circuits. Dense SELECT synthesis is costed, not called efficient.

No objective values are loaded into the circuit. The residual is constructed
from the LCHS propagator, the quote observation matrix and reference prices.
"""
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, UnitaryGate
from qiskit.quantum_info import Statevector

def contraction_dilation(B):
    """Halmos unitary dilation; B is a contraction. Ancilla is the high bit."""
    U,s,Vh=sla.svd(B); s=np.clip(s,0,1)
    left=(U*np.sqrt(np.maximum(0,1-s*s)))@U.conj().T
    V=Vh.conj().T; right=(V*np.sqrt(np.maximum(0,1-s*s)))@V.conj().T
    full=np.block([[B,left],[right,-B.conj().T]])
    if not np.allclose(full.conj().T@full,np.eye(len(full)),atol=2e-10):
        raise AssertionError("Dilation is not unitary")
    return full

def select_condition(circuit,gate,controls,targets,value):
    """Little-endian integer convention, explicitly tested by statevector readback."""
    if not controls:
        circuit.append(gate,targets); return
    for j,bit in enumerate(controls):
        if not ((value>>j)&1): circuit.x(bit)
    circuit.append(gate.control(len(controls)),controls+targets)
    for j,bit in enumerate(controls):
        if not ((value>>j)&1): circuit.x(bit)

def reflection_success(width,ancillas):
    """2 P_success - I, with success defined by all listed ancillas being zero."""
    qc=QuantumCircuit(width,name="2P-I"); qc.global_phase=np.pi
    for a in ancillas: qc.x(a)
    if len(ancillas)==1: qc.z(ancillas[0])
    else: qc.mcp(np.pi,ancillas[:-1],ancillas[-1])
    for a in ancillas: qc.x(a)
    return qc

def build_residual_circuit(cfg):
    n=cfg["circuit_n"]; m=cfg["circuit_m"]; g=make_grid(n,2)
    N=g["N"]; d=2*m; D=2**d; J=cfg["circuit_terms"]; b=int(np.log2(J))
    if 2**b!=J: raise ValueError("Circuit LCU count must be a power of two")
    k,w,meta=lchs_weights("beta",J,4.); alpha=float(np.abs(w).sum())
    dt=float(TIMES[0]); c=g["cbar"]
    design=list(range(d)); work=list(range(d,d+n)); kr=list(range(d+n,d+n+b))
    wa=d+n+b; branch=wa+1; width=branch+1
    # Three distinct physical quote locations (two when N=4).
    qk=np.array([.85,1.1]) if N==4 else np.array([.8,1.,1.25])
    Wfull=interpolation(g["K"],qk); Wsmall=Wfull[:,1:-1]
    W=np.zeros((N,N)); W[:len(qk)]=Wsmall
    gammaW=max(1.,sla.norm(W,2)); UW=contraction_dilation(W/gammaW)
    xi_star=np.array([.7,.25]); exact=sla.expm(-dt*assemble(g,xi_star))@g["v0"]
    ref=Wsmall@exact; y=np.zeros(N); y[:len(ref)]=ref
    vnorm=sla.norm(g["v0"]); ynorm=sla.norm(y)
    a=vnorm*np.exp(c*dt)*alpha*gammaW; scale=a+ynorm
    prep=StatePreparation(np.sqrt(np.abs(w)/alpha))
    forward=QuantumCircuit(width-1,name="LCHS-W")
    forward.append(StatePreparation(g["v0"]/vnorm),work)
    forward.append(prep,kr)
    values=[]; losses=[]; forward_errors=[]
    for index in range(D):
        xi=np.array([index%(2**m),index//(2**m)])/(2**m-1)
        A=assemble(g,xi); L=(A+A.T)/2+c*np.eye(N); H=-.5j*(A-A.T)
        approx=np.zeros((N,N),complex)
        for j in range(J):
            U=sla.expm(-1j*dt*(H+k[j]*L))
            approx+=w[j]*U
            gate=UnitaryGate(np.exp(1j*np.angle(w[j]))*U,label="exp(-i(H+kL)t)")
            # controls ordered design then LCU index
            select_condition(forward,gate,design+kr,work,index+(j<<d))
        pv=np.exp(c*dt)*approx@g["v0"]
        loss=float(np.linalg.norm(W@pv-y)**2)
        values.append(xi); losses.append(loss)
        forward_errors.append(float(np.max(np.abs(pv-sla.expm(-dt*A)@g["v0"]))))
    forward.append(prep.inverse(),kr)
    forward.append(UnitaryGate(UW,label="quote W dilation"),work+[wa])
    V=QuantumCircuit(width,name="Residual")
    branchprep=StatePreparation([np.sqrt(a/scale),np.sqrt(ynorm/scale)])
    V.append(branchprep,[branch])
    V.x(branch); V.append(forward.to_gate().control(1),[branch]+list(range(width-1))); V.x(branch)
    V.append(StatePreparation(y/ynorm).control(1),[branch]+work)
    V.z(branch)
    V.append(branchprep.inverse(),[branch])
    return dict(V=V,g=g,d=d,D=D,N=N,values=np.array(values),losses=np.array(losses),
                scale=scale,ancillas=kr+[wa,branch],meta=meta,forward_errors=forward_errors,
                gammaW=gammaW,lcualpha=alpha,dt=dt)

def run_quantum_checks(cfg):
    out=cfg["out"]; print("Constructing explicit coherent residual circuit...")
    t=time.perf_counter(); built=build_residual_circuit(cfg)
    V=built["V"]; d=built["d"]; D=built["D"]; N=built["N"]; width=V.num_qubits
    initial=QuantumCircuit(width)
    for bit in range(d): initial.h(bit)
    residual_state=Statevector.from_instruction(initial.compose(V)).data.reshape(-1,D)
    probabilities=np.sum(np.abs(residual_state[:N])**2,axis=0)*D
    expected=built["losses"]/built["scale"]**2
    residual_error=float(np.max(np.abs(probabilities-expected)))
    assert residual_error<2e-8,"Residual circuit does not match matrix calculation"
    # V^dagger (2P-I) V is block-encoded by clean workspace. Add one
    # Hadamard-test flag so the clean block is exactly F / scale^2.
    R=reflection_success(width,built["ancillas"])
    U=QuantumCircuit(width,name="2F-1")
    U.compose(V,inplace=True); U.compose(R,inplace=True); U.compose(V.inverse(),inplace=True)
    F=QuantumCircuit(width+1,name="Objective BE")
    flag=width; F.h(flag); F.append(U.to_gate().control(1),[flag]+list(range(width))); F.h(flag)
    ini=QuantumCircuit(width+1)
    for bit in range(d): ini.h(bit)
    state=Statevector.from_instruction(ini.compose(F)).data
    observed=state[:D]*np.sqrt(D)
    oracle_error=float(np.max(np.abs(observed-expected)))
    assert oracle_error<3e-8,"Objective block-encoding mismatch"
    print("Transpiling dense small-instance oracle into u/cx gates...")
    compiled=transpile(F,basis_gates=["u","cx"],optimization_level=0,seed_transpiler=17)
    counts=compiled.count_ops()
    resource=dict(n=cfg["circuit_n"],M=2,m=cfg["circuit_m"],
                  design_points=D,lcu_terms=cfg["circuit_terms"],qubits=F.num_qubits,
                  depth=compiled.depth(),cx=int(counts.get("cx",0)),u=int(counts.get("u",0)),
                  residual_probability_error=residual_error,objective_block_error=oracle_error,
                  classical_build_validation_s=time.perf_counter()-t,
                  scale=built["scale"],alpha_lcu=built["lcualpha"],gamma_W=built["gammaW"],
                  cost_scope="Explicit dense design-enumerated SELECT; not asymptotic efficient synthesis",
                  gate_set="u,cx, all-to-all connectivity, no fault-tolerant compilation",hardware_execution=False)
    save_json(out/"quantum_resources.json",resource)
    frame=pd.DataFrame(dict(index=np.arange(D),xi1=built["values"][:,0],xi2=built["values"][:,1],
                  physical_loss=built["losses"],normalised_loss=expected,
                  circuit_probability=probabilities,circuit_block_real=observed.real,
                  circuit_block_imag=observed.imag,forward_price_error=built["forward_errors"]))
    frame.to_csv(out/"quantum_objective.csv",index=False)
    # Save a readable circuit diagram of the high-level validated composition.
    (out/"diagnostics"/"circuit.txt").write_text(str(F.draw(output="text",fold=100)))
    print("Quantum statevector checks PASS:",resource)
    return frame,resource

def run_qhd(cfg,landscape):
    """Classical simulation of QHD on the verified discrete objective landscape.
    The landscape is classically tabulated. This is not a full coherent run.
    """
    size=int(np.sqrt(len(landscape))); F=landscape.physical_loss.to_numpy()
    scale=max(float(np.ptp(F)),1e-12); V=(F-F.min())/scale
    h=1/(size-1)
    lap=np.diag(np.r_[1.,np.full(size-2,2.),1.])-np.diag(np.ones(size-1),1)-np.diag(np.ones(size-1),-1)
    kinetic=(np.kron(np.eye(size),lap)+np.kron(lap,np.eye(size)))/(2*h*h)
    eig,Q=sla.eigh(kinetic); psi=np.ones(size*size,complex)/size
    steps=cfg["qhd_steps"]; total=5.; dt=total/steps; records=[]
    target=F<=F.min()+.01*scale
    def kinetic_step(p,coef): return Q@(np.exp(-1j*coef*eig)*(Q.T@p))
    for j in range(steps+1):
        if j%max(1,steps//100)==0 or j==steps:
            p=np.abs(psi)**2
            records.append(dict(step=j,time=j*dt,expected_loss=float(p@F),
                                best_grid_loss=float(F.min()),near_optimal_probability=float(p@target),norm=float(p.sum())))
        if j==steps: break
        s=(j+.5)*dt
        psi=kinetic_step(psi,.5*dt*np.exp(-s))
        psi*=np.exp(-1j*dt*np.exp(s)*V)
        psi=kinetic_step(psi,.5*dt*np.exp(-s))
    assert abs(np.sum(np.abs(psi)**2)-1)<1e-8
    result=pd.DataFrame(records); result.to_csv(cfg["out"]/"qhd_trajectory.csv",index=False)
    final=landscape.copy(); final["probability"]=np.abs(psi)**2
    final.to_csv(cfg["out"]/"qhd_distribution.csv",index=False)
    save_json(cfg["out"]/"qhd_status.json",dict(scope="Classically tabulated landscape; split-step QHD simulation",
            strong_convexity_verified=False,global_continuous_optimum_verified=False,
            reference="Exact enumeration of this finite design grid",schedule="exp(-s) kinetic + exp(s) normalised potential"))
    return result
