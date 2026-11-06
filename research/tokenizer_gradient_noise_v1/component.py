"""CPU FP64 streaming moments of complete fixed-weight B8 gradient draws."""
import math
import torch

class Moments:
    def __init__(self):self.count=0;self.means={};self.m2={};self.schema=None
    def update(self,gradients):
        if type(gradients) is not dict or not gradients or any(type(k) is not str or not k for k in gradients):raise ValueError('Complete named gradient dictionary required')
        schema={}
        # Prevalidate the ENTIRE input before destination mutation.
        for name,g in gradients.items():
            if type(g) is not torch.Tensor or g.device.type!='cpu' or g.dtype not in (torch.float32,torch.float64) or g.layout!=torch.strided or g.numel()==0 or not bool(torch.isfinite(g).all()):raise ValueError('Finite nonempty CPU float gradients required')
            schema[name]=(tuple(g.shape),g.dtype)
        if self.schema is not None and schema!=self.schema:raise ValueError('Complete stable parameter keys/shape/dtype required')
        n=self.count+1
        if self.schema is None:
            means={k:v.detach().to(dtype=torch.float64,copy=True) for k,v in gradients.items()};self.schema=schema;self.means=means;self.m2={k:0. for k in means}
        else:
            for name,g in gradients.items():
                x=g.detach().to(dtype=torch.float64);delta=x-self.means[name];self.means[name].add_(delta,alpha=1/n)
                self.m2[name]+=float((delta*(x-self.means[name])).sum())
        self.count=n
    def report(self):
        if self.count<2:raise ValueError('At least two complete gradient draws required')
        per={}
        for name,mu in self.means.items():
            norm=float(mu.square().sum());cov=self.m2[name]/(self.count-1)
            if not math.isfinite(norm+cov) or cov<0:raise ValueError('Finite nonnegative empirical trace covariance required')
            per[name]=dict(mean_gradient_squared_norm=norm,trace_sample_covariance=cov,signal_squared_unbiased=norm-cov/self.count)
        norm=math.fsum(v['mean_gradient_squared_norm'] for v in per.values());cov=math.fsum(v['trace_sample_covariance'] for v in per.values());signal=norm-cov/self.count
        return dict(draws=self.count,parameter_statistics=per,mean_gradient_squared_norm=norm,trace_sample_covariance=cov,signal_squared_unbiased=signal,
                    ratio_in_independent_B8_draw_units=(cov/signal if signal>0 else None),positive_signal_estimate=signal>0,
                    scope='Raw unpreconditioned gradients at fixed weights. Each unit is an independent paired512 B8 draw,not an IID token. Ratio is biased and may be undefined;not optimal batch size,Muon/Adam update noise or a quality/generalization prediction.')
