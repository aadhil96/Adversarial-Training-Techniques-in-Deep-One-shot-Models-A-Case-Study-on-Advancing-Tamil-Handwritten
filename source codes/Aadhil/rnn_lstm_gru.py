import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


#input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001  

input_size = 28
sequence_length = 28
num_layers = 2

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))])


#MNIST
train_dataset = torchvision.datasets.MNIST(root='./data',train=True,
                transform = transform, download = True)

test_dataset = torchvision.datasets.MNIST(root='./data',train=False,
                transform = transform)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size=batch_size,shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size=batch_size,shuffle=True) 


examples = iter(train_loader)
samples , labels = examples.next()

class RNN(nn.Module):
    def __init__(self, input_size , hidden_size, num_layers, num_classes):
        super(RNN,self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
        #x-> (batch_size,seq,input_size)
        self.fc = nn.Linear(hidden_size,num_classes  )
        
        
    def forward(self,x):
        h0 = torch.zeros(self.num_layers,x.size(0) , self.hidden_size)

        out,_ = self.rnn(x,h0)
        #out : batch_size , sequence_length
        # out (N , 28,128)
        out = out[:,-1,:]
        # out (N , 128)
        out = self.fc(out)
        return out

model = RNN(input_size,hidden_size,num_layers, num_classes) 

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters() , lr=learning_rate)

#training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i,(images , labels) in enumerate(train_loader):
        # 100 , 1 , 28 , 28
        # 100 , 28 , 28
        images = images.reshape(-1, sequence_length,input_size)
        
        #forward
        outputs = model(images)
        loss = criterion(outputs,labels)
        
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'epoch {epoch+1} / {num_epochs} , step {i+1}/{n_total_steps}, loss={loss.item():.4f}')
            

#test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images , labels in test_loader:
        images = images.reshape(-1,sequence_length,input_size)
        outputs = model(images)
        #value , index 
        _, predictions = torch.max(outputs,1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
        
    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}')

   