
import torch
from torch import nn
from functools import partial
import pdb

@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w, scales


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w, scales


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t.reshape(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t, scales


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t.contiguous().view(-1, t_shape[-1])  ##contiguous added
    scales = t.abs().max()
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t, scales


class W8A8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, act_quant='per_token', weight_quant='per_tensor', quantize_output=False): ## weight_quant added
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_quant_name='per_tensor'

        self.register_buffer('weight', torch.randn(self.out_features,
                                                   self.in_features, dtype=torch.float16, requires_grad=False))
        
        if bias:
            self.register_buffer('bias', torch.zeros(
                (1, self.out_features), dtype=torch.float16, requires_grad=False))
        else:
            self.register_buffer('bias', None)

        if act_quant == 'per_token':
            self.act_quant_name = 'per_token'
            self.act_quant = partial(
                quantize_activation_per_token_absmax, n_bits=8)
        elif act_quant == 'per_tensor':
            self.act_quant_name = 'per_tensor'
            self.act_quant = partial(
                quantize_activation_per_tensor_absmax, n_bits=8)
        else:
            raise ValueError(f'Invalid act_quant: {act_quant}')

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = 'None'
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(W8A8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x, _ = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)    
        if self.output_quant_name== "None":
            q_y = self.output_quant(y)
        else:
            q_y, _ = self.output_quant(y)    
        return q_y

    @staticmethod
    def from_float(module, weight_quant='per_channel', act_quant='per_token', quantize_output=False):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8A8Linear(
            module.in_features, module.out_features, module.bias is not None, act_quant=act_quant, weight_quant=weight_quant, quantize_output=quantize_output)
        if weight_quant == 'per_channel':
            new_module.weight, _ = quantize_weight_per_channel_absmax(
                module.weight, n_bits=8)  # use 8-bit integer for weight
        elif weight_quant == 'per_tensor':
            new_module.weight, _ = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=8)
        else:
            raise ValueError(f'Invalid weight_quant: {weight_quant}')
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias    
        return new_module

    def __repr__(self):
        return f'W8A8Linear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})'



class NoisyW8A8Linear(W8A8Linear):
    def __init__(self, in_features, out_features, bias=True, act_quant='per_token', quantize_output=False, err_prob=0.0, accumulation_bitw=32):
        super().__init__(in_features,out_features,bias,act_quant,quantize_output)
        assert isinstance(err_prob, list) or isinstance(err_prob, float)
        self.err_prob=err_prob
        self.accumulation_bitw = accumulation_bitw
        self.w_scales=None

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = 'None'
            self.output_quant = lambda x: x
    
    @torch.no_grad()
    def inject_error(self,y, w_scales, a_scales, err_prob):
        y_not_quantized=y
        y_div_a_scales=y/a_scales
        result = y_div_a_scales/w_scales.view(1,1,-1)
        # result=y.to(torch.float32)/(w_scales*a_scales)  ## integer of y
        result=result.round().to(torch.int32)
        result_injected=result
        
        #You can change flip_bit to any bits(0~31)
        flip_bit=30
        err=torch.tensor([2**flip_bit],dtype=torch.int32).to(result.device)
        prob_tensor=torch.full(result.shape, err_prob).to(result.device)
        mask=torch.bernoulli(prob_tensor).bool().to(result.device)

        result_injected[mask]=torch.bitwise_xor(result[mask],err)
        
        result_injected=result_injected.to(torch.float32)*a_scales*w_scales.view(1,1,-1)
        # result_injected=result_injected.to(torch.float32)*a_scales*w_scales
        result_injected=result_injected.to(y.dtype)
        y_not_quantized[mask]=result_injected[mask]

        # if not (y_not_quantized == y).all():
        #     import pdb; pdb.set_trace()
        return y_not_quantized

    @torch.no_grad()
    def forward(self, x):  
        q_x, x_scales = self.act_quant(x) ## q_x is multiplied by scale
        y = torch.functional.F.linear(q_x, self.weight, bias=None)
        y_for_quant= torch.functional.F.linear(q_x, self.weight, bias=self.bias)
        y_injected=self.inject_error(y, self.w_scales, x_scales, self.err_prob)
        if self.bias is not None:
            y_injected=y_injected + self.bias

        if self.output_quant_name== "None":
            q_y = self.output_quant(y_injected)  
            # clipping=y_for_quant.abs().max()

            # q_y=torch.where((q_y>clipping)|(q_y<-clipping), torch.zeros_like(q_y), q_y)
            # q_y = torch.clamp(q_y,-clipping,clipping) ## avoid overflowing of float16
        else:
            _, out_scale = self.output_quant(y_for_quant)  
            q_y=torch.clamp(torch.round(y_injected/out_scale),-127,127)*out_scale ## quant according to out_scale
        return q_y

    @staticmethod
    def from_float(module, weight_quant='per_channel', act_quant='per_token', quantize_output=False, err_prob=0.0,accumulation_width=32):
        assert isinstance(module, torch.nn.Linear)
        new_module = NoisyW8A8Linear(
            module.in_features, module.out_features, module.bias is not None, act_quant=act_quant, quantize_output=quantize_output,err_prob=err_prob,accumulation_bitw=accumulation_width)
        if weight_quant == 'per_channel':
            new_module.weight, new_module.w_scales = quantize_weight_per_channel_absmax(
                module.weight, n_bits=8)  # use 8-bit integer for weight
        elif weight_quant == 'per_tensor':
            new_module.weight, new_module.w_scales = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=8)
        else:
            raise ValueError(f'Invalid weight_quant: {weight_quant}')
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f'NoisyW8A8Linear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name}, err_prob={self.err_prob})'


class W8A8BMM(nn.Module):
    def __init__(self, act_quant='per_token',quantize_output=False):
        super().__init__()
        
        if act_quant=='per_token':
            self.act_quant_name='per_token'
            self.act_quant=partial(quantize_activation_per_token_absmax,n_bits=8)
        elif act_quant=='per_tensor':
            self.act_quant_name='per_tensor'
            self.act_quant=partial(quantize_activation_per_tensor_absmax,n_bits=8)
        else:
            raise ValueError(f'Invalid act_qunant: {act_quant}')
        
        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = 'None'
            self.output_quant = lambda x: x

    @torch.no_grad()
    def forward(self, input1, input2):
        # pdb.set_trace()
        q_input1, _ = self.act_quant(input1)
        q_input2, _ = self.act_quant(input2)
        y = torch.bmm(q_input1, q_input2)
        
        if self.output_quant_name== "None":
            q_y = self.output_quant(y)
        else:
            q_y, _ = self.output_quant(y)
        return q_y

    def __repr__(self):
        return f'W8A8BMM(act_quant={self.act_quant_name}, output_quant={self.output_quant_name})'




class NoisyW8A8BMM(W8A8BMM):
    def __init__(self, act_quant='per_token', quantize_output=False, err_prob=0.0, accumulation_bitw=32):
        super().__init__(act_quant,quantize_output)
        assert isinstance(err_prob, list) or isinstance(err_prob, float)
        self.err_prob=err_prob
        self.accumulation_bitw = accumulation_bitw
        
    @torch.no_grad()
    def inject_error(self,y, w_scales, a_scales, err_prob):
        y_not_quantized=y
        y_div_a_scales=y/a_scales
        result = y_div_a_scales/w_scales.view(1,1,-1)
        #result=y.to(torch.float32)/(w_scales*a_scales)  ## integer of y
        result=result.round().to(torch.int32)
        result_injected=result
        flip_bit=30
        err=torch.tensor([2**flip_bit],dtype=torch.int32).to(result.device)
        prob_tensor=torch.full(result.shape, err_prob).to(result.device)
        mask=torch.bernoulli(prob_tensor).bool().to(result.device)
        result_injected[mask]=torch.bitwise_xor(result[mask],err)
        result_injected=result_injected.to(torch.float32)*a_scales*w_scales.view(1,1,-1)
        result_injected=result_injected.to(y.dtype)
        y_not_quantized[mask]=result_injected[mask]

        return y_not_quantized
    
    @torch.no_grad()
    def forward(self, input1, input2):
        q_input1, input1_scale = self.act_quant(input1)
        q_input2, input2_scale = self.act_quant(input2)
        y = torch.bmm(q_input1, q_input2)
        y_clone=y.clone()
        y_injected=self.inject_error(y_clone,input1_scale,input2_scale,self.err_prob)
        # y_injected = y
        
        if self.output_quant_name== "None":
            q_y = self.output_quant(y_injected)
            # q_y = torch.clamp(q_y,-32768,32768) ## avoid overfitting of float16
        else:
            _, out_scale = self.output_quant(y)
            q_y=torch.clamp(torch.round(y_injected/out_scale),-127,127)*out_scale ## quant according to out_scale    
            # q_y, _ = self.output_quant(y)
        return q_y
    
    def __repr__(self):
        return f'NoisyW8A8BMM(act_quant={self.act_quant_name}, output_quant={self.output_quant_name}, err_prob={self.err_prob})'

class layer_norm_without_outlier(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, percentage=0.1):
        super().__init__(normalized_shape, eps, elementwise_affine)
        self.percentage = percentage

    @torch.no_grad()
    def forward(self,x):
        upper=x.quantile(self.percentage)
        lower=x.quantile(1-self.percentage)
        mask=(x<lower)|(x>upper)
        medium_elements=x.clone()
        medium_elements[mask]=0.
        mu=medium_elements.mean(dim=1,keepdim=True)
        sigma2=medium_elements.var(dim=1,keepdim=True)
        x_normalized=(x-mu)/torch.sqrt(sigma2+self.eps)*self.weight+self.bias
        return x_normalized
    
    @staticmethod
    def from_float(module,percentage):
        new_module=layer_norm_without_outlier(normalized_shape=module.normalized_shape, 
                                              eps=module.eps, 
                                              elementwise_affine=module.elementwise_affine,
                                              percentage=percentage).to('cuda')
        new_module.weight=module.weight
        new_module.bias=module.bias
        return new_module
    
    def __repr__(self):
        return f'layer_norm_without_outlier(percentage={self.percentage})'
    
class W8A8MatMul(nn.Module):
    def __init__(self, act_quant='per_token',quantize_output=False):
        super().__init__()
        
        if act_quant=='per_token':
            self.act_quant_name='per_token'
            self.act_quant=partial(quantize_activation_per_token_absmax,n_bits=8)
        elif act_quant=='per_tensor':
            self.act_quant_name='per_tensor'
            self.act_quant=partial(quantize_activation_per_tensor_absmax,n_bits=8)
        else:
            raise ValueError(f'Invalid act_qunant: {act_quant}')
        
        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = 'None'
            self.output_quant = lambda x: x

    @torch.no_grad()
    def forward(self, input1, input2):
        # pdb.set_trace()
        q_input1, _ = self.act_quant(input1)
        q_input2, _ = self.act_quant(input2)
        y = torch.matmul(q_input1, q_input2)
        
        if self.output_quant_name== "None":
            q_y = self.output_quant(y)
        else:
            q_y, _ = self.output_quant(y)
        return q_y

    def __repr__(self):
        return f'W8A8MatMul(act_quant={self.act_quant_name}, output_quant={self.output_quant_name})'
    

class NoisyW8A8MatMul(W8A8MatMul):
    def __init__(self, act_quant='per_token', quantize_output=False, err_prob=0.0, accumulation_bitw=32):
        super().__init__(act_quant,quantize_output)
        assert isinstance(err_prob, list) or isinstance(err_prob, float)
        self.err_prob=err_prob
        self.accumulation_bitw = accumulation_bitw
        
    @torch.no_grad()
    def inject_error(self,y, w_scales, a_scales, err_prob):
        y_not_quantized=y
        scale=torch.matmul(w_scales,a_scales.permute(0,1,3,2))
        result=y/scale
       
        #result=y.to(torch.float32)/(w_scales*a_scales)  ## integer of y
        result=result.round().to(torch.int32)
        result_injected=result
        flip_bit=30
        err=torch.tensor([2**flip_bit],dtype=torch.int32).to(result.device)
        prob_tensor=torch.full(result.shape, err_prob).to(result.device)
        mask=torch.bernoulli(prob_tensor).bool().to(result.device)
        result_injected[mask]=torch.bitwise_xor(result[mask],err)
        result_injected=result_injected.to(torch.float32)*scale
        result_injected=result_injected.to(y.dtype)
        y_not_quantized[mask]=result_injected[mask]
        
        return y_not_quantized
    
    @torch.no_grad()
    def forward(self, input1, input2):
        q_input1, input1_scale = self.act_quant(input1)
        q_input2, input2_scale = self.act_quant(input2.permute(0,1,3,2))
        y = torch.matmul(q_input1, q_input2.permute(0,1,3,2))
        y_clone=y.clone()
        y_injected=self.inject_error(y_clone,input1_scale,input2_scale,self.err_prob)
        # y_injected = y
        
        if self.output_quant_name== "None":
            q_y = self.output_quant(y_injected)
            # q_y = torch.clamp(q_y,-32768,32768) ## avoid overfitting of float16
        else:
            _, out_scale = self.output_quant(y)
            q_y=torch.clamp(torch.round(y_injected/out_scale),-127,127)*out_scale ## quant according to out_scale    
            # q_y, _ = self.output_quant(y)
        return q_y
    
    def __repr__(self):
        return f'NoisyW8A8MatMul(act_quant={self.act_quant_name}, output_quant={self.output_quant_name}, err_prob={self.err_prob})'
    
