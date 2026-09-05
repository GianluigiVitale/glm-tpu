import sys, numpy as np, time
sys.path.insert(0,'.'); from ref_common import *
from safetensors.numpy import load_file
t0=time.time(); tok=tokens(); T=len(tok); x=embed_rows(tok); h=rmsnorm(x,weight('model.layers.0.input_layernorm.weight'),RMS_EPS); np.save('l0_x.npy',x); np.save('l0_h.npy',h)
L='model.layers.0.self_attn.'
Wqa=weight(L+'q_a_proj.weight'); wqa_n=weight(L+'q_a_layernorm.weight')
q_resid=rmsnorm(h@Wqa.T, wqa_n, LORA_EPS)                       # [T,2048]
Wqb_i=weight(L+'indexer.wq_b.weight'); Wk=weight(L+'indexer.wk.weight'); kw=weight(L+'indexer.k_norm.weight'); kb=weight(L+'indexer.k_norm.bias'); Wwp=weight(L+'indexer.weights_proj.weight')
pos=8155
q=(q_resid[pos]@Wqb_i.T).reshape(32,128)                          # [32,128]
k=layernorm(h@Wk.T,kw,kb,KNORM_EPS)                               # [T,128]
w=(h[pos]@Wwp.T)*(32**-0.5)                                       # [32]
cos,sin=rope_angles(np.arange(T),64); cq,sq=rope_angles([pos],64)
oracle='/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/8k/greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z/oracle/dsa_events.safetensors'
d=load_file(oracle); oc=int(d['valid_counts'][0,0]); O=set(d['selected_positions'][0,0,:oc].tolist()); osc=d['selected_scores'][0,0,:oc].astype(np.float64); opos=d['selected_positions'][0,0,:oc]
print("oracle event0 count",oc,"cutoff",osc[-1])
for name,rq,rk in (("interleaved",rope_interleaved,rope_interleaved),("half",rope_half,rope_half)):
    qq=q.copy(); qq[:,:64]=rq(q[:,:64],cq[0],sq[0])   # rotate the single-position query, broadcast over heads
    kk=k.copy(); kk[:,:64]=rk(k[:,:64],cos,sin)
    R=w@np.maximum((qq@kk.T)*128**-0.5,0)                          # [T]
    order=np.argsort(-R,kind='stable'); sel=set(order[:2048].tolist())
    rO=R[opos]; dd=osc-rO
    print("%-12s swaps vs oracle event0: %d ; oracle-R aligned max %.5f mean %.6f std %.6f ; R cutoff %.5f"%(name,len(sel^O)//2,np.abs(dd).max(),dd.mean(),dd.std(),R[order[2047]]))
np.save('l0_q_resid.npy',q_resid); print("done",time.time()-t0)
