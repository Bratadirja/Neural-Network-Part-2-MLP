import numpy as np

def fowardpass(input, weight, bias, activation = 'linier') :

    w_sum = np.dot(input,weight) + bias

    if activation is 'relu' :
        #ReLU Activation f(x) = max(0,x)
        act = np.maximum(w_sum,0)

    else :
        #Linier Activation
        act = w_sum

    return act

# Pre-Trained Weights & Biases After Training
W_H = np.array([[0.00192761, -0.78845304, 0.30310717, 0.44131625, 0.32792646, -0.02451803, 1.43445349, -1.12972116]])
B_H = np.array([-0.02657719, -1.15885878, -0.79183501, -0.33550513, -0.23438406, -0.25078532, 0.22305705, 0.80253315])

W_O = np.array([[-0.77540326], [ 0.5030424 ], [ 0.37374797], [-0.20287184], [-0.35956827], [-0.54576212], [ 1.04326093], [ 0.8857621 ]])
B_O = np.array([ 0.04351173])

# Initialize Input Data
inputs = np.array([[-2],[0],[2]])

#Output of Hidden Layer
h_out = fowardpass(inputs, W_H, B_H, 'relu')

print("Hiden Layer Output (ReLU)")
print("=========================")
print(h_out,"\n")

# Output of Output Layer
o_out = fowardpass(h_out, W_O, B_O, 'linier')

print("Ouput Layer Output (linier)")
print("===========================")
print(o_out,"\n")
