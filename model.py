import torch
import torch.nn as nn

class NeuralNet(nn.Module):

    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet, self).__init__()
        #3개의 선형레이어 만들기
        self.l1=nn.Linear(input_size,hidden_size) #input_size와 num_classes는 고정값, hidden_size는 변경할 수 있음
        self.l2=nn.Linear(hidden_size,hidden_size) 
        self.l3=nn.Linear(hidden_size,num_classes) 
        self.relu=nn.ReLU() #relu 활셩화 함수 사용
    
    def forward(self,x): #정방향
        out=self.l1(x)
        out=self.relu(out)
        out=self.l2(out)
        out=self.relu(out)
        out=self.l3(out)
        #no activation and no softmax
        return out      


