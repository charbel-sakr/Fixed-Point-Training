import theano
import theano.tensor as T
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class QuantGradOp(theano.Op):
    def __init__(self,B,GMAX):
        self.B=B
        self.GMAX=GMAX
        super(QuantGradOp,self).__init__()

    def make_node(self,x):
        x = theano.tensor.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])
    def perform(self,node,inputs,output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = x
    def infer_shape(self, node, i0_shapes):
        return i0_shapes
    def grad(self,inputs,output_grads):
        B=self.B
        GMAX=self.GMAX
        in_grad = GMAX*T.minimum(1.0-T.pow(2.0,1.0-B),T.round((output_grads[0]/GMAX)*T.pow(2.0,B-1.0),mode="half_away_from_zero")*T.pow(2.0,1.0-B))
        return [in_grad]

B =[7.0,10.0,10.0,11.0,11.0,11.0,11.0,10.0,11.0,11.0,10.0,9.0,9.0,10.0,10.0,11.0,11.0,11.0,13.0,15.0]
DR =[2.0*.000244140625,2.0*0.0001220703125,2.0*0.0001220703125,2.0*0.0001220703125,2.0*0.0001220703125,2.0*0.0001220703125,2.0*0.0001220703125,2.0*0.0001220703125,2.0*0.000244140625,2.0*0.000244140625,2.0*0.0001220703125,2.0*0.0001220703125,2.0*0.0001220703125,2.0*0.000244140625,2.0*0.000244140625,2.0*0.000244140625,2.0*0.0001220703125,2.0*0.0001220703125,2.0*0.0001220703125,2.0*0.03125]
a=0.0
s = 1.0
quantizeGradL1 = QuantGradOp(B[0]+a,s*DR[0])
quantizeGradL2 = QuantGradOp(B[1]+a,s*DR[1])
quantizeGradL3 = QuantGradOp(B[2]+a,s*DR[2])
quantizeGradL4 = QuantGradOp(B[3]+a,s*DR[3])
quantizeGradL5 = QuantGradOp(B[4]+a,s*DR[4])
quantizeGradL6 = QuantGradOp(B[5]+a,s*DR[5])
quantizeGradL7 = QuantGradOp(B[6]+a,s*DR[6])
quantizeGradL8 = QuantGradOp(B[7]+a,s*DR[7])
quantizeGradL9 = QuantGradOp(B[8]+a,s*DR[8])
quantizeGradL10 = QuantGradOp(B[9]+a,s*DR[9])
quantizeGradL11 = QuantGradOp(B[10]+a,s*DR[10])
quantizeGradL12 = QuantGradOp(B[11]+a,s*DR[11])
quantizeGradL13 = QuantGradOp(B[12]+a,s*DR[12])
quantizeGradL14 = QuantGradOp(B[13]+a,s*DR[13])
quantizeGradL15 = QuantGradOp(B[14]+a,s*DR[14])
quantizeGradL16 = QuantGradOp(B[15]+a,s*DR[15])
quantizeGradL17 = QuantGradOp(B[16]+a,s*DR[16])
quantizeGradL18 = QuantGradOp(B[17]+a,s*DR[17])
quantizeGradL19 = QuantGradOp(B[18]+a,s*DR[18])
quantizeGradL20 = QuantGradOp(B[19]+a,s*DR[19])
