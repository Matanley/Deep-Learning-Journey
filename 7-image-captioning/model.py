import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
    
class DecoderRNN(nn.Module):
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, dropout=0.5):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Define the embedding layers
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # Define the LSTM layer with droputout
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, features, captions):
        
        # add one more dimension for features as LSTM input
        features = features.view(-1, 1, self.embed_size)
        
        captions = self.embed(captions)
        
        # Leave out the end token and concatenate features and captions as inputs
        inputs = torch.cat((features, captions[:, :-1,:]), dim=1)
        
        # Pass forward the LSTM layer
        lstm_output, lstm_hidden = self.lstm(inputs)
        
        # Pass forward the fully connected layer to turn the output into vectors in the size (batch_size, sequence_length, vocab_size)
        output = self.fc(lstm_output)
        
        output_scores = self.softmax(output)
        return output_scores
   

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        # Initialize a list to store the generated results
        generated = []
        
        # pass forward the inputs to get the predicted index
        lstm_output, lstm_hidden = self.lstm(inputs)
        output = self.fc(lstm_output)
        score = self.softmax(output)
        predicted_index = torch.argmax(score, dim=-1).item()
       
        generated.append(predicted_index)
        
        # loop through the max_len with the previous step output as the current step input
        for i in range(max_len - 1):
            
            next_input = self.embed(torch.LongTensor([[predicted_index]]).cuda())
            lstm_output, lstm_hidden = self.lstm(next_input, lstm_hidden)
            output = self.fc(lstm_output)
            score = self.softmax(output)
            predicted_index = torch.argmax(score, dim=-1).item()
            
            generated.append(predicted_index)
        
        return generated
            
        