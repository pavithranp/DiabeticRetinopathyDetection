
"""Model code"""

batch_size = 10
epochs = 20
learning_rate = 0.0001

"""class MyModel(k.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv0 = k.layers.Conv2D(8, (3, 3),strides =2,activation='relu')
        self.conv1 = k.layers.Conv2D(16, (3, 3),strides =2,activation='relu')
        self.flatten = k.layers.Flatten()
        self.dense0 =  k.layers.Dense(128,activation='relu')
        self.dense1 =  k.layers.Dense(64,activation='relu')
        self.dense2a =  k.layers.Dense(5,activation='softmax',name="Grade")
        self.dense2b = k.layers.Dense(3,activation='softmax',name="Risk")
        

        # Layer definition

    def call(self, inputs, training=False):
        output = self.conv0(inputs)
        output = self.conv1(output)
        output = self.flatten(output)        
        output = self.dense0(output) 
        output = self.dense1(output)
        output1 = self.dense2a(output)
        output2 = self.dense2b(output)
        #output = k.layers.Concatenate([output1,output2])
        # Call layers appropriately to implement a forward pass
        return output1,output2"""


inputs = k.layers.Input(shape=(256,256,3))

# a layer instance is callable on a tensor, and returns a tensor

conv2d1 = k.layers.Conv2D(16, (3, 3),strides=(1, 1), activation='relu')(inputs)
BN1 = k.layers.BatchNormalization()(conv2d1)
LLR1 = k.layers.LeakyReLU()(BN1)

conv2d2 = k.layers.Conv2D(16, (3, 3),strides=(2, 2), activation='relu')(LLR1)
BN2 = k.layers.BatchNormalization()(conv2d2)
LLR2 = k.layers.LeakyReLU()(BN2)

AP = k.layers.AveragePooling2D(pool_size=8)(LLR2)

flatten = k.layers.Flatten()(AP)

dense0 = k.layers.Dense(128, activation='relu')(flatten)
BN3 = k.layers.BatchNormalization()(dense0)
LLR3 = k.layers.LeakyReLU()(BN3)


dense1 = k.layers.Dense(64, activation='relu')(flatten)
BN4 = k.layers.BatchNormalization()(dense1)
LLR4 = k.layers.LeakyReLU()(BN4)

output1 = k.layers.Dense(5, activation='softmax',name='Grade')(LLR3)

output2 = k.layers.Dense(3, activation='softmax',name='Risk')(LLR4)
#output = k.layers.Concatenate([output1,output2])

# This creates a model that includes
# the Input layer and three Dense layers
model = k.models.Model(inputs=inputs, outputs=[output1,output2])

opt = k.optimizers.RMSprop(learning_rate)
losses = {
	"Grade": "sparse_categorical_crossentropy",
	"Risk": "sparse_categorical_crossentropy",
}
model.compile(optimizer=opt ,loss=losses, metrics= ['accuracy'])
#model.build(input_shape=(32,256,256,3))

model.summary()
#model.output_shape()

#inp,grade,risk = train_ds.take(413)
model.fit(train_ds,validation_data=validation_ds, epochs = 40)
