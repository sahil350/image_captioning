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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # embeddings layer
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer
        self.lstm = nn.LSTM (
                            input_size = embed_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            bias = True,
                            batch_first = True,
                            dropout = 0.1,
                            bidirectional = False
                        )
        # the linear layer that maps the hidden state output dimension 
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        
    def init_hidden(self, batch_size):
        ''' At the start of training, we need to initialize a hidden state;
           there will be none because the hidden state is formed based on perviously seen data.
           So, this function defines a hidden state with all zeroes and of a specified size.'''
        # The axes dimensions are (n_layers, batch_size, hidden_dim)
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))
    
    def forward(self, features, captions):
        ''' Define the feedforward behavior of the model.'''
        
        # discard <end> word
        captions = captions[:,:-1]
        
        # initialize hidden_state
        batch_size = features.shape[0] # features shape is batch_sizeXembed_size
        hidden = self.init_hidden(batch_size)
        
        # create embedded word vectors for each word in a caption
        embeds = self.word_embeddings(captions)
        
        # stack features and caption
        embeds = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hiddent state
        lstm_out, self.hidden = self.lstm(embeds, hidden)
        
        # get the outputs
        outputs = self.linear(lstm_out)
        
        return outputs
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        # init output
        output = []
        
        # initialize hidden state
        batch_size = inputs.shape[0]
        hidden = self.init_hidden(batch_size)
        
        while True:
            lstm_out, hidden = self.lstm(inputs, hidden)
            
            # get ouput
            outputs = self.linear(lstm_out)
            
            # squeeze output
            outputs = outputs.squeeze(1)
            
            # predict most likely next word
            _, most_likely = torch.max(outputs, dim = 1)
            
            # append the predicted word to ouput
            output.append(most_likely.cpu().numpy()[0].item())
            
            # break if <end> predicted
            if most_likely == 1:
                break
            
            # embed predicted word to input
            inputs = self.word_embeddings(most_likely)
            
            # unsqueeze inputs
            inputs = inputs.unsqueeze(1)
            
        return output