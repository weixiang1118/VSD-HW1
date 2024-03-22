from copy import deepcopy
import torch.nn as nn
import torch
from typing import Tuple
from typing import List
import numpy as np

def copy_model(model: nn.Module) -> nn.Module:
    result = deepcopy(model)

    # Copy over the extra metadata we've collected which copy.deepcopy doesn't capture
    if hasattr(model, 'input_activations'):
        result.input_activations = deepcopy(model.input_activations)

    for result_layer, original_layer in zip(result.modules(), model.modules()):
        if isinstance(result_layer, nn.Conv2d) or isinstance(result_layer, nn.Linear):
            if hasattr(original_layer.weight, 'scale'):
                result_layer.weight.scale = deepcopy(
                    original_layer.weight.scale)

        if hasattr(original_layer, 'inAct'):
            result_layer.inAct = deepcopy(original_layer.inAct)
        if hasattr(original_layer, 'outAct'):
            result_layer.outAct = deepcopy(original_layer.outAct)
        if hasattr(original_layer, 'output_scale'):
            result_layer.output_scale = deepcopy(original_layer.output_scale)

    return result

def quantized_weights(weights: torch.Tensor) -> Tuple[torch.Tensor, float]:
    '''
    Quantize the weights so that all values are integers between -128 and 127. 2^8
    Use the total range when deciding just what factors to scale the float32
    values by.

    Parameters:
    weights (Tensor): The unquantized weights

    Returns:
    (Tensor, float): A tuple with the following elements:
        * The weights in quantized form, where every value is an integer between -128 and 127.
          The "dtype" will still be "float", but the values themselves should all be integers.
        * The scaling factor that your weights were divided by.
          This value does not need to be an 8-bit integer.
    '''
    # TODO use handout lec04 page40 method2
    rmax= torch.max(torch.abs(weights))
    #rmin = torch.min(weights)
    B = 8
    scale = (2*rmax)/((2**B)-1)
    #Z = torch.round(-rmin/scale) = 0
    weights = torch.round(weights/scale)
    weights = weights.clamp(-128,127)
    return weights, scale

from typing import List

class NetQuantized(nn.Module):
    def __init__(self, net_with_weights_quantized: nn.Module):
        super(NetQuantized, self).__init__()

        net_init = copy_model(net_with_weights_quantized)

        self.conv1 = net_init.conv1
        self.maxpool2 = net_init.maxpool2
        self.conv3 = net_init.conv3
        self.maxpool4 = net_init.maxpool4
        self.conv5 = net_init.conv5
        self.fc6 = net_init.fc6
        self.output = net_init.output

        for layer in self.conv1, self.conv3, self.conv5, self.fc6, self.output:
            def pre_hook(l, x):
                x = x[0]
                if (x < -128).any() or (x > 127).any():
                    raise Exception("Input to {} layer is out of bounds for an 8-bit signed integer".format(l.__class__.__name__))
                if (x != x.round()).any():
                    raise Exception("Input to {} layer has non-integer values".format(l.__class__.__name__))
            layer.register_forward_pre_hook(pre_hook)

        # Calculate the scaling factor for the initial input to the CNN
        self.input_activations = net_with_weights_quantized.conv1.inAct
        self.input_scale = NetQuantized.quantize_initial_input(self.input_activations)

        # Calculate the output scaling factors for all the layers of the CNN
        self.preceding_layer_scales = []
        for layer in self.conv1, self.conv3, self.conv5, self.fc6, self.output:
            layer.output_scale, n_o = NetQuantized.quantize_activations(layer.outAct, layer[0].weight.scale, self.input_scale, self.preceding_layer_scales)
            self.preceding_layer_scales.append(n_o)

    @staticmethod
    def quantize_initial_input(pixels: np.ndarray) -> float:
        '''
        Calculate a scaling factor for the images that are input to the first layer of the CNN.

        Parameters:
        pixels (ndarray): The values of all the pixels which were part of the input image during training

        Returns:
        float: A scaling factor that the input should be "multiplied" by before being fed into the first layer. 
               In other words, the return value should be 1/s_I instead of s_I. Check description in 2.4 for more details. 
               This value does not need to be an 8-bit integer.
        '''

        # TODO
        rmax = torch.max(torch.abs(pixels))
        B=8
        s_initial_input = (2*rmax)/((2**B) -1)
        return 1.0 / s_initial_input

    @staticmethod
    def quantize_activations(activations: np.ndarray, s_w: float, s_initial_input: float, s_o_prev: List[float]) -> Tuple[float, float]:
        '''
        Calculate a scaling factor to multiply the output of a layer.

        Parameters:
        activations (ndarray): The values of all the pixels which have been output by this layer during training
        s_w (float): The scale by which the weights of this layer were divided as part of the "quantize_weights" function you wrote earlier
        s_initial_input (float): The scale by which the initial input to the neural network was multiplied. Note that it is 1/s_I instead of s_I.
        s_o_prev ([float]): A list of floats, where each represents the scaling factor of the output matrix of the previous layers.

        Returns:
        (M, s_o) A tuple with the following elements:
        M (float): A scaling factor that the layer output should be multiplied by before being fed into the next layer. This value does not need to be an 8-bit integer.
        s_o (float): The scaling factor of the output matrix of the current layer. This is used to calculate the next layer's output scale.
        '''
        # TODO
        rmax = torch.max(torch.abs(activations))
        B = 8
        s_o = (2*rmax) / ((2**B) -1)
        #print(s_o)
        if len(s_o_prev) == 0:
            M = s_w / (s_initial_input * s_o)
        else :
            M = (s_w* s_o_prev[-1]) / (s_o)
        #print(M)
        return M, s_o

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Since input_scale is 128 and all output_scales are less than 1, we should keep input_scale as it is and tranform output_scale to
        round(1/output_scale) to ease the verilog implementaion.

        Also, the normalized input images is a matrix with lots of floating numbers. We can transform x*input_scale to
        input_scale/round(1/x)

        To not implement rounding in verilog, we use floor when doing other calculations with input/output_scale.
        '''
        # To make sure that the outputs of each layer are integers between -128 and 127, you may need to use the following functions:
        #   * torch.Tensor.round
        #   * torch.Tensor.floor
        #   * torch.clamp
        # TODO
        x = torch.floor(self.input_scale / torch.round(1/x))
        x = x.clamp(-128,127)
        
        x = self.conv1(x)
        
        #re-quantization
        scale = torch.round((self.conv1.output_scale)*(2**16))
        x = torch.floor((x * scale).to(torch.int) >> 16)
        x = x.to(torch.float)
        x = x.clamp(-128,127)
        
        x = self.maxpool2(x)
        
        '''scale = torch.round((self.maxpool2.output_scale)*(2**16))
        x = torch.floor((x * scale).to(torch.int) >> 16)
        x = x.to(torch.float)
        x = x.clamp(-128,127)'''
        
        x = self.conv3(x)
        
        scale = torch.round( self.conv3.output_scale *(2**16))
        x = torch.floor((x * scale).to(torch.int) >> 16)
        x = x.to(torch.float)
        x = x.clamp(-128,127)
        
        x = self.maxpool4(x)
        
        '''scale = torch.round( self.maxpool4.output_scale *(2**16))
        x = torch.floor((x * scale).to(torch.int) >> 16)
        x = x.to(torch.float)
        x = x.clamp(-128,127)'''
        
        x = self.conv5(x)
        
        scale = torch.round( self.conv5.output_scale*(2**16) )
        x = torch.floor((x * scale).to(torch.int) >> 16)
        x = x.to(torch.float)
        x = x.clamp(-128,127)
        x = torch.flatten(x, 1)
        
        x = self.fc6(x)
        
        scale = torch.round( self.fc6.output_scale *(2**16) )
        x = torch.floor((x * scale).to(torch.int) >> 16)
        x = x.to(torch.float)
        x = x.clamp(-128,127)
        
        x = self.output(x)
        
        scale = torch.round( self.output.output_scale *(2**16) )
        x = torch.floor((x * scale).to(torch.int) >> 16)
        x = x.to(torch.float)
        x = x.clamp(-128,127)
        
        return x
    
    
class NetQuantizedWithBias(NetQuantized):
    def __init__(self, net_with_weights_quantized: nn.Module):
        super(NetQuantizedWithBias, self).__init__(net_with_weights_quantized)

        self.output[0].bias.data = NetQuantizedWithBias.quantized_bias(
            self.output[0].bias.data,
            self.output[0].weight.scale,
            self.preceding_layer_scales
        )

        if (self.output[0].bias.data < -2147483648).any() or (self.output[0].bias.data > 2147483647).any():
            raise Exception("Bias has values which are out of bounds for an 32-bit signed integer")
        if (self.output[0].bias.data != self.output[0].bias.data.round()).any():
            raise Exception("Bias has non-integer values")

    @staticmethod
    def quantized_bias(bias: torch.Tensor, s_w: float, s_o_prev: List[float]) -> torch.Tensor:
        '''
        Quantize the bias so that all values are integers between -2147483648 and 2147483647.

        Parameters:
        bias (Tensor): The floating point values of the bias
        s_w (float): The scale by which the weights of this layer were divided
        s_o_prev ([float]): A list of floats, where each represents the scaling factor of the output matrix of the previous layers.

        Returns:
        Tensor: The bias in quantized form, where every value is an integer between -2147483648 and 2147483647.
                The "dtype" will still be "float", but the values themselves should all be integers.
        '''

        # TODO
        # --------------------------------------------------------------------
        # The quantization method are similar to quantize output activation
        # --------------------------------------------------------------------
        
        B=31
        rmax= torch.max(torch.abs(bias))
        scale = (2*rmax)/((2**8)-1)
        bias = torch.round(bias/scale)
        bias = bias.clamp(-(2**B),(2**B)-1)
        
        return bias
    

def float_to_fixed_scale(act_scalesDict, weight_scalesDict, outputBias_float):
    scalesDict = {
        'quant': 0,
        'conv1.conv': 0,
        'conv3.conv': 0,
        'conv5.conv': 0,
        'fc6.fc': 0,
        'output.fc': 0
    }
    outputBias_fixed = []
    # TODO
    B=16
    #print(outputBias_float)
    for layerName in scalesDict.keys():
        if layerName == 'quant':

            scalesDict[layerName] = int(np.round(1/act_scalesDict[layerName]).item())
  
        elif layerName == 'conv1.conv':
            
            Ml = int(np.round(((weight_scalesDict[layerName]*act_scalesDict['quant'])/act_scalesDict[layerName])*(2**B)).item())
            scalesDict[layerName] = Ml
            
        elif layerName == 'conv3.conv':
            
            Ml = int(np.round((weight_scalesDict[layerName]*act_scalesDict['conv1.conv']/act_scalesDict[layerName])*(2**B)).item())
            scalesDict[layerName] = Ml
            
        elif layerName == 'conv5.conv':

            Ml = int(np.round((weight_scalesDict[layerName]*act_scalesDict['conv3.conv']/act_scalesDict[layerName])*(2**B)).item())
            scalesDict[layerName] = Ml
            
        elif layerName == 'fc6.fc':

            Ml = int(np.round((weight_scalesDict[layerName]*act_scalesDict['conv5.conv']/act_scalesDict[layerName])*(2**B)).item())
            scalesDict[layerName] = Ml

        elif layerName == 'output.fc':

            Ml = int(np.round((weight_scalesDict[layerName]*act_scalesDict['fc6.fc']/act_scalesDict[layerName])*(2**B)).item())
            scalesDict[layerName] = Ml
            outputBias_fixed.append(np.round(outputBias_float/(weight_scalesDict[layerName]*act_scalesDict['fc6.fc'])).astype(int))
            outputBias_fixed = outputBias_fixed[0]

    return scalesDict, outputBias_fixed