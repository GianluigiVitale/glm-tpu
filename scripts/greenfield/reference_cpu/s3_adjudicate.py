import numpy as np, ml_dtypes, json, hashlib
from safetensors.numpy import load_file
bf=lambda a: np.asarray(a).astype(np.float32).astype(ml_dtypes.bfloat16).astype(np.float64)
ref=np.load('l1_indexer_inputs.npz'); R=np.load('l1_event1_R.npy')          # math reference (FP64) at event 1
li=np.load('../layer1_internals/internals.model_layers_1_self_attn_attn.position8155.proc0.npz')
lq=li['query'].astype(np.float64); lw=li['head_weights'].astype(np.float64); lck=li['current_key'].astype(np.float64)
lnh=li['normalized_hidden'].view(ml_dtypes.bfloat16).astype(np.float64); lqa=li['q_a_state'].view(ml_dtypes.bfloat16).astype(np.float64)
lkeys=load_file('../l1cache/prompt_index_cache.safetensors')['prompt_index_key_bfloat16_bits'].view(ml_dtypes.bfloat16).astype(np.float64)
def rel(a,b):  # legacy a vs reference b
    d=a-b; return {"max_abs":float(np.abs(d).max()),"mean":float(d.mean()),"std":float(d.std()),"rel_rms":float(np.sqrt((d*d).mean())/np.sqrt((b*b).mean())),"ref_rms":float(np.sqrt((b*b).mean()))}
out={"validation_legacy_vs_reference":{
  "normalized_hidden_8155":rel(lnh,ref['normalized_hidden_8155']),
  "q_a_state_8155":rel(lqa,ref['q_a_state_8155']),
  "query_8155":rel(lq,ref['query_8155']),
  "head_weights_8155":rel(lw,ref['head_weights_8155']),
  "current_key_8155":rel(lck,ref['keys'][8155]),
  "prompt_keys_0_8154":rel(lkeys,ref['keys'][:8155])}}
print(json.dumps(out,indent=1))
# event-1 rows
oracle='/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/8k/greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z/oracle/'
d=load_file(oracle+'dsa_events.safetensors'); oc=int(d['valid_counts'][0,1]); oi=d['selected_positions'][0,1,:oc]; os_=d['selected_scores'][0,1,:oc].astype(np.float64)
e=np.load('../ws32_0827/runner.rank0.npz'); ec=int(e['dsa_selected_valid_counts'][0,1,0]); ei=e['dsa_selected_positions'][0,1,0,:ec]; es=e['dsa_selected_scores'][0,1,0,:ec].astype(np.float64)
E=set(ei.tolist()); O=set(oi.tolist()); A=sorted(E&O); ed=dict(zip(ei.tolist(),es)); od=dict(zip(oi.tolist(),os_))
order=np.argsort(-R,kind='stable'); Rset=set(order[:2048].tolist()); cR=R[order[2047]]
dO=np.array([od[p] for p in A])-R[A]; dE=np.array([ed[p] for p in A])-R[A]
kappa=2.0; n=len(A)
eps=float(np.abs(dO).max()); capE=float(np.abs(dE).max()); stdO=float(dO.std()); stdE=float(dE.std())
mo=float(dO.mean()); me=float(dE.mean()); s=float(dE.std())
cE=es[ec-1]; cO=os_[oc-1]
sym=sorted(E^O)
explained=[bool(abs(R[p]-cR)<=eps) for p in sym]   # reference-band explanation
band=int((np.abs(R-cR)<=eps).sum())
item3={"eps_event":eps,"engine_max_abs":capE,"engine_max_ok":bool(capE<=kappa*eps),"std_o":stdO,"std_e":stdE,"engine_std_ok":bool(stdE<=kappa*stdO),
       "sym_diff":sym,"sym_diff_ref_score_minus_ref_cutoff":[float(R[p]-cR) for p in sym],"all_within_eps_of_ref_cutoff":all(explained),"ref_band_size":band,"sym_ok_count":bool(len(sym)<=band),
       "engine_cutoff":float(cE),"oracle_cutoff":float(cO),"reference_cutoff":float(cR),
       "reference_set_swaps_vs_oracle":len(Rset^O)//2,"reference_set_swaps_vs_engine":len(Rset^E)//2}
item4={"m_o":mo,"m_e":me,"s":s,"n":n,"bound":kappa*abs(mo)+3*s/np.sqrt(n),"ok":bool(abs(me)<=kappa*abs(mo)+3*s/np.sqrt(n))}
out["event1_vs_math_reference"]={"oracle_minus_R":{"max_abs":eps,"mean":mo,"std":stdO},"engine_minus_R":{"max_abs":capE,"mean":me,"std":stdE},"item3":item3,"item4":item4,"kappa":kappa}
out["verdict"]={"item3_pass":bool(item3["engine_max_ok"] and item3["engine_std_ok"] and item3["all_within_eps_of_ref_cutoff"] and item3["sym_ok_count"]),"item4_pass":bool(item4["ok"])}
print(json.dumps(out["event1_vs_math_reference"],indent=1)); print("VERDICT",out["verdict"])
json.dump(out,open('s3_adjudication.json','w'),indent=1)
for f in ('l1_indexer_inputs.npz','l1_event1_R.npy','l0_out_x2.npy','l0_sel.npy'):
    print(f, hashlib.sha256(open(f,'rb').read()).hexdigest())
