from .deepsense_params import *
import torch
import torch.nn as nn


class DeepSense:
    
    def __init__(self) -> None:
        pass
    
    def conv2d_layer(self, inputs, filter_size, kernel_size, padding, activation=None):  #il faut retourner des nn.sequential avec la conv puis l'activation
        
        return nn.Conv2d(in_channels=inputs,
                        filters=filter_size,
                        kernel_size=[1, kernel_size],
                        stride=(1, 1),
                        padding=padding,
                        activation=activation,
                    )

    def dense_layer(self, inputs, num_units, name, reuse, activation=None):
        output = nn.Linear(
                        in_features=inputs,
                        out_features=num_units,
                        activation=activation,
                        name=name,
                        reuse=reuse
                    )
        return output

    def dropout_layer(self, inputs, keep_prob, name, is_conv=False):
        if is_conv:
            channels = torch.shape(inputs)[-1]
            return nn.Dropout(
                            inputs,
                            p=keep_prob,
                            name=name,
                            noise_shape=[
                                self.batch_size, 1, 1, channels
                            ]
                        )
        else:
            return nn.Dropout(
                        inputs,
                        p=keep_prob,
                        name=name)