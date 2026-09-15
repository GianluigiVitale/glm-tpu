"""FP64 reference: full layer-0 (dense) forward for positions 0..8155, then layer-1 indexer inputs.
Independent of the greenfield engine.  torch.float64 on CPU for heavy matmuls."""
import sys, time, numpy as np, torch
sys.path.insert(0,'.'); from ref_common import *
torch.set_num_threads(200); torch.set_default_dtype(torch.float64)
t0=time.time()
def log(*a): print("[%7.1fs]"%(time.time()-t0),*a,flush=True)
tok=tokens(); T=len(tok); x=torch.from_numpy(np.load('l0_x.npy')); h=torch.from_numpy(np.load('l0_h.npy')); log("loaded x,h",T)
L='model.layers.0.self_attn.'
W=lambda n: torch.from_numpy(weight(n))
# ---- attention projections
q_resid=torch.from_numpy(np.load('l0_q_resid.npy'))                      # [T,2048]
Wqb=W(L+'q_b_proj.weight'); q=(q_resid@Wqb.T).view(T,64,256); del Wqb
q_nope=q[:,:,:192].contiguous(); q_rot=q[:,:,192:].contiguous(); del q
Wkva=W(L+'kv_a_proj_with_mqa.weight'); kv=h@Wkva.T; del Wkva            # [T,576]
kva_n=W(L+'kv_a_layernorm.weight'); k_lat=kv[:,:512]; k_rot=kv[:,512:].contiguous()
k_lat=k_lat*torch.rsqrt((k_lat*k_lat).mean(-1,keepdim=True)+LORA_EPS)*kva_n
Wkvb=W(L+'kv_b_proj.weight'); kvb=(k_lat@Wkvb.T).view(T,64,448); del Wkvb
k_nope=kvb[:,:,:192].contiguous(); v=kvb[:,:,192:].contiguous(); del kvb
log("projections done")
pos=np.arange(T); cos,sin=rope_angles(pos,64); cos=torch.from_numpy(cos); sin=torch.from_numpy(sin)
def rope_il(xx,c,s):  # xx [...,T,64] pairs interleaved; c,s [T,32]
    x1=xx[...,0::2]; x2=xx[...,1::2]; out=torch.empty_like(xx); out[...,0::2]=x1*c-x2*s; out[...,1::2]=x2*c+x1*s; return out
q_rot=rope_il(q_rot.transpose(0,1),cos,sin).transpose(0,1).contiguous()   # [T,64,64]
k_rot=rope_il(k_rot,cos,sin)                                             # [T,64]
# ---- indexer (layer 0) selection for every position
Wqbi=W(L+'indexer.wq_b.weight'); qi=(q_resid@Wqbi.T).view(T,32,128); del Wqbi
qi=qi.clone(); qi[:,:,:64]=rope_il(qi[:,:,:64].transpose(0,1),cos,sin).transpose(0,1)
Wk=W(L+'indexer.wk.weight'); kw=W(L+'indexer.k_norm.weight'); kb=W(L+'indexer.k_norm.bias')
ki=h@Wk.T; mu=ki.mean(-1,keepdim=True); var=((ki-mu)**2).mean(-1,keepdim=True); ki=(ki-mu)/torch.sqrt(var+KNORM_EPS)*kw+kb
ki=ki.clone(); ki[:,:64]=rope_il(ki[:,:64],cos,sin)
Wwp=W(L+'indexer.weights_proj.weight'); wi=(h@Wwp.T)*(32**-0.5)          # [T,32]
sel=torch.full((T,2048),-1,dtype=torch.int64); selcount=torch.zeros(T,dtype=torch.int64)
B=256
for s0 in range(0,T,B):
    s1=min(T,s0+B); qb=qi[s0:s1]                                          # [b,32,128]
    sc=torch.einsum('bhd,td->bht',qb,ki)*(128**-0.5); sc=torch.relu(sc)   # [b,32,T]
    isc=torch.einsum('bh,bht->bt',wi[s0:s1],sc)                           # [b,T]
    tpos=torch.arange(T)[None,:]; qpos=torch.arange(s0,s1)[:,None]; isc=isc.masked_fill(tpos>qpos,float('-inf'))
    k=min(2048,s1)  # rows with fewer causal keys select all
    top=torch.topk(isc,k=2048 if s1>2048 else s1,dim=-1,largest=True,sorted=True)
    for i in range(s1-s0):
        n=min(2048,s0+i+1); idx=top.indices[i,:n]; idx=idx[torch.isfinite(top.values[i,:n])]
        sel[s0+i,:len(idx)]=idx; selcount[s0+i]=len(idx)
    if s0%2048==0: log("indexer block",s0)
np.save('l0_sel.npy',sel.numpy()); np.save('l0_isc_8155.npy',isc[-1].numpy() if s1==T else np.zeros(1))
log("layer-0 selection done")
# ---- sparse MLA attention
scale=256**-0.5; attn=torch.empty(T,64,256)
Bq=32
for s0 in range(0,T,Bq):
    s1=min(T,s0+Bq); b=s1-s0; idx=sel[s0:s1].clone(); n=selcount[s0:s1]
    mask=torch.arange(2048)[None,:]<n[:,None]; idx[~mask]=0
    Kn=k_nope[idx]; Kr=k_rot[idx]; Vs=v[idx]                              # [b,2048,64,192],[b,2048,64],[b,2048,64,256]
    logits=torch.einsum('bhd,bshd->bhs',q_nope[s0:s1],Kn)+torch.einsum('bhd,bsd->bhs',q_rot[s0:s1],Kr)
    logits=logits*scale; logits=logits.masked_fill(~mask[:,None,:],float('-inf'))
    p=torch.softmax(logits,dim=-1); attn[s0:s1]=torch.einsum('bhs,bshd->bhd',p,Vs)
    if s0%1024==0: log("attention block",s0)
del Kn,Kr,Vs,k_nope,v
Wo=W(L+'o_proj.weight'); ao=attn.reshape(T,-1)@Wo.T; del Wo; x1=x+ao; log("attention+o_proj done")
# ---- dense MLP
M='model.layers.0.mlp.'
h2=x1*torch.rsqrt((x1*x1).mean(-1,keepdim=True)+RMS_EPS)*W('model.layers.0.post_attention_layernorm.weight')
Wg=W(M+'gate_proj.weight'); g=h2@Wg.T; del Wg
Wu=W(M+'up_proj.weight'); u=h2@Wu.T; del Wu
act=g*torch.sigmoid(g)*u; del g,u
Wd=W(M+'down_proj.weight'); x2=x1+act@Wd.T; del Wd,act
np.save('l0_out_x2.npy',x2.numpy()); np.save('l0_attn_out_8155.npy',ao[8155].numpy()); log("layer 0 complete; x2 saved")
# ---- layer-1 indexer inputs
L1='model.layers.1.self_attn.'
h1=x2*torch.rsqrt((x2*x2).mean(-1,keepdim=True)+RMS_EPS)*W('model.layers.1.input_layernorm.weight')
Wqa1=W(L1+'q_a_proj.weight'); qa1=h1@Wqa1.T; del Wqa1
qres1=qa1*torch.rsqrt((qa1*qa1).mean(-1,keepdim=True)+LORA_EPS)*W(L1+'q_a_layernorm.weight')
Wqbi1=W(L1+'indexer.wq_b.weight'); q1=(qres1[8155]@Wqbi1.T).view(32,128).clone(); q1[:,:64]=rope_il(q1[:,:64],cos[8155],sin[8155])
Wk1=W(L1+'indexer.wk.weight'); kw1=W(L1+'indexer.k_norm.weight'); kb1=W(L1+'indexer.k_norm.bias')
k1=h1@Wk1.T; mu=k1.mean(-1,keepdim=True); var=((k1-mu)**2).mean(-1,keepdim=True); k1=(k1-mu)/torch.sqrt(var+KNORM_EPS)*kw1+kb1
k1=k1.clone(); k1[:,:64]=rope_il(k1[:,:64],cos,sin)
w1=(h1[8155]@W(L1+'indexer.weights_proj.weight').T)*(32**-0.5)
np.savez('l1_indexer_inputs.npz', normalized_hidden_8155=h1[8155].numpy(), q_a_state_8155=qres1[8155].numpy(), query_8155=q1.numpy(), head_weights_8155=w1.numpy(), keys=k1.numpy())
R1=w1@torch.relu((q1@k1.T)*(128**-0.5)); np.save('l1_event1_R.npy',R1.numpy())
log("layer-1 indexer inputs + event-1 reference row saved")
