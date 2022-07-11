import json
from chattest import tokenize,stem,bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

from model import NeuralNet

with open('intents.json','r') as f:
    intents=json.load(f)

all_words=[]
tags=[]
xy=[]
for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w=tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))

ignore_words=['?','!','.',',']
all_words=[stem(w) for w in all_words if w not in ignore_words] #형태소분석 적용
all_words=sorted(set(all_words)) #중복단어 X
tags=sorted(set(tags))

print(tags)

#훈련데이터
x_train=[]
y_train=[]
for (pattern_sentence,tag) in xy: #pattenr_sentence : 이미 토큰화 된 문장?
    bag=bag_of_words(pattern_sentence,all_words)
    x_train.append(bag) #훈련데이터에 추가
    
    label=tags.index(tag) #tag 인덱스를 얻은 후 y데이터에 넣음
    y_train.append(label) #CrossEntropyLoss(교차엔트로피손실?)

x_train=np.array(x_train)
y_train=np.array(y_train)

#새 채팅 데이터 세트
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples=len(x_train)
        self.x_data=x_train
        self.y_data=y_train

    #dataset[idx]
    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

#HyperParameters
batch_size=8
hidden_size=8
output_size=len(tags)
input_size=len(x_train[0])
learning_rate=0.001
num_epochs=1000 #훈련횟수?
# print(input_size,len(all_words))
# print(output_size,tags)

dataset=ChatDataset()
train_loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=2)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#모델 생성
model=NeuralNet(input_size,hidden_size,output_size).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(num_epochs):
    for (words,labels) in train_loader:
        words=words.to(device)
        labels=labels.to(device)

        #forward
        outputs=model(words)
        loss=criterion(outputs,labels)

        #backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch +1) % 100 ==0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')