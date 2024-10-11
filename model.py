import torch   # set the interpreter /.venv/bin/python3 to resolve the autocomplete issue
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        Initializes the input embedding layer.
        This class handles the input embeddings, which convert token indices into dense vectors.
        It's an important part of how the Transformer processes input text.

        Args:
            d_model (int): The dimensionality of the embeddings and model (e.g., 512).
            vocab_size (int): The number of unique tokens (words) in the vocabulary.

        Attributes:
            embedding (nn.Embedding): Layer that converts tokens into dense vectors of size `d_model`.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        Forward pass for the input embedding layer.
            
        Args:
            x (Tensor): A tensor of token indices.

        Returns:
            Tensor: The embedding for each token scaled by sqrt(d_model).
        """

        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        Initializes the positional encoding layer.
        
        This class adds positional encodings to the input embeddings to allow the model to capture
        the relative positions of tokens in a sequence. It uses a sinusoidal function to generate
        the encodings and applies dropout to regularize the model.

        Args:
            d_model (int): The size of the embedding vectors (same as in InputEmbeddings).
            seq_len (int): The maximum sequence length (number of tokens in a sentence).
            dropout (float): Dropout rate applied to prevent overfitting.

        Attributes:
            pe (Tensor): A buffer containing the positional encodings, of shape (1, seq_len, d_model).
            dropout (nn.Dropout): Dropout layer.
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrixof Shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)   # (1, seq_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Adds positional encodings to the input tensor.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
            Tensor: Input tensor with positional encodings added.
        """
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps: float = 10**-6) -> None:
        """
        Initializes the layer normalization.
        
        Layer normalization is used to stabilize and speed up the training process by normalizing
        the inputs across the features (or dimensions) of the input tensor.
        
        Args:
            eps (float, optional): A small value to avoid division by zero, defaults to 1e-6.

        Attributes:
            alpha (nn.Parameter): A learnable scaling factor initialized to 1.
            bias (nn.Parameter): A learnable bias (offset) initialized to 0.
        """

        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter


    def forward(self, x):
        """
        Applies layer normalization to the input tensor.
        
        Args:
            x (Tensor): Input tensor to normalize, typically of shape (batch_size, seq_len, d_model).
        
        Returns:
            Tensor: Normalized tensor with the same shape as the input.
        """

        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """
        Initializes the feed-forward block.
        
        This block applies two linear transformations with a ReLU activation in between,
        and a dropout layer for regularization.

        Args:
            d_model (int): The dimensionality of the input and output vectors.
            d_ff (int): The dimensionality of the hidden layer.
            dropout (float): Dropout rate to apply after the ReLU activation.

        Attributes:
            linear_1 (nn.Linear): First linear transformation layer.
            linear_2 (nn.Linear): Second linear transformation layer.
            dropout (nn.Dropout): Dropout layer to apply between the transformations.
        """

        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 and B2
    
    def forward(self, x):
        """
        Applies the feed-forward transformations to the input tensor.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
            Tensor: Output tensor with the same shape as the input.
        """

        # (Batch, Seq_len, d_model) --> (Batch, Seq_Len, d_ff) --> (Batch, Seq_Len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        """
        Initializes the multi-head attention block.

        This block implements the multi-head self-attention mechanism used in Transformer models.
        It allows the model to focus on different parts of the input sequence simultaneously,
        which improves the model's ability to capture relationships between tokens at different
        positions in the sequence.

        Args:
            d_model (int): The dimensionality of the input and output vectors.
            h (int): The number of attention heads.
            dropout (float): Dropout rate applied to the attention scores for regularization.

        Attributes:
            d_model (int): Dimensionality of the input (e.g., 512 or 768).
            h (int): Number of attention heads.
            d_k (int): Dimensionality of each attention head (d_model // h).
            w_q (nn.Linear): Linear layer to project input into query vectors.
            w_k (nn.Linear): Linear layer to project input into key vectors.
            w_v (nn.Linear): Linear layer to project input into value vectors.
            w_o (nn.Linear): Linear layer for output transformation after attention.
            dropout (nn.Dropout): Dropout layer applied to attention scores.
        """

        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by the number of heads (h)."

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)   # Wq
        self.w_k = nn.Linear(d_model, d_model)   # Wk
        self.w_v = nn.Linear(d_model, d_model)   # Wv

        self.w_o = nn.Linear(d_model, d_model)   # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Computes the scaled dot-product attention for the input queries, keys, and values.

        Args:
            query (Tensor): Query tensor of shape (batch_size, h, seq_len, d_k).
            key (Tensor): Key tensor of shape (batch_size, h, seq_len, d_k).
            value (Tensor): Value tensor of shape (batch_size, h, seq_len, d_k).
            mask (Tensor or None): Optional tensor to mask certain positions in the sequence.
            dropout (nn.Dropout): Dropout layer applied to attention scores for regularization.

        Returns:
            Tensor: The weighted sum of the values, shape (batch_size, h, seq_len, d_k).
            Tensor: Attention scores (probabilities) of shape (batch_size, h, seq_len, seq_len).
        """

        d_k = query.shape[-1]

        # (Batch, h, Seq_Len, d_k)  -->  (Batch, h, Seq_Len, Seq_Len)
        # Compute scaled dot-product attention scores
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        # Convert scores to probabilities using softmax
        attention_scores = attention_scores.softmax(dim = -1)  # (Batch, h, Seq_Len, Seq_Len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Weighted sum of values using the attention scores     
        return (attention_scores @value), attention_scores

    
    def forward(self, q, k, v, mask):
        """
        Forward pass through the multi-head attention block.

        Args:
            q (Tensor): Query input tensor of shape (batch_size, seq_len, d_model).
            k (Tensor): Key input tensor of shape (batch_size, seq_len, d_model).
            v (Tensor): Value input tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor or None): Optional mask tensor to prevent attention to certain positions.

        Returns:
            Tensor: Output of the multi-head attention block, shape (batch_size, seq_len, d_model).
        """

        # Project input to query, key, and value vectors
        query = self.w_q(q)     # (Batch, Seq_Len, d_model)  -->  (Batch, Seq_Len, d_model) 
        key = self.w_k(k)       # (Batch, Seq_Len, d_model)  -->  (Batch, Seq_Len, d_model) 
        value = self.w_v(v)     # (Batch, Seq_Len, d_model)  -->  (Batch, Seq_Len, d_model) 

        # Reshape to (batch_size, seq_len, h, d_k) and transpose to (batch_size, h, seq_len, d_k)
        # (Batch, Seq_Len, d_model)  -->  (Batch, Seq_Len, h, d_k)  -->  (Batch, h, Seq_Len, d_k) 
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        # (Batch, Seq_Len, d_model)  -->  (Batch, Seq_Len, h, d_k)  -->  (Batch, h, Seq_Len, d_k) 
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        # (Batch, Seq_Len, d_model)  -->  (Batch, Seq_Len, h, d_k)  -->  (Batch, h, Seq_Len, d_k) 
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Compute attention using the static method
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Concatenate attention outputs from all heads
        # (Batch, h, Seq_Len, d_k) --> (Batch, Seq_Len, h, d_k) --> (Batch, Seq_Len, d_model) 
        x = x.transpose(1, 2).contiguous().view(x.shape[0],  -1, self.h * self.d_k)

        # Apply the output linear transformation
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model) 
        return self.w_o(x)
    

class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:
        """
        Implements a residual connection followed by layer normalization and dropout.

        The residual connection helps in training deep networks by adding the input (`x`) directly to the output of 
        the sublayer. Layer normalization is applied to stabilize the training process, and dropout is used for regularization.

        Args:
            dropout (float): Dropout rate to be applied after the sublayer.

        Attributes:
            dropout (nn.Dropout): Dropout layer to regularize the sublayer.
            norm (LayerNormalization): Layer normalization to stabilize the output.
        """

        super().__init__()
        self.dropout = nn.Dropout(dropout)  # Apply dropout to prevent overfitting
        self.norm = LayerNormalization(features)    # Layer normalization for stabilizing the learning process

    def forward(self, x, sublayer):
        """
        Forward pass through the residual connection.

        Args:
            x (Tensor): Input tensor to the residual connection.
            sublayer (Callable): The sublayer (e.g., self-attention or feed-forward block) to apply normalization and dropout.

        Returns:
            Tensor: The output of the residual connection, which is the sum of the normalized sublayer output and the original input `x`.
        """
        # Apply layer normalization, pass through the sublayer, and apply dropout
        return x + self.dropout(sublayer(self.norm(x)))  # Add original input (residual connection)


class EncoderBlock(nn.Module):
    """
    The role of the self-attention block and feed-forward network, both wrapped in residual connections.
    The forward method has comments that clarify the flow through the self-attention and feed-forward blocks
    with their respective residual connections.
    """

    def __init__(self,
                 features: int, 
                 self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        """
        Represents a single block in the encoder of the Transformer.

        Each encoder block consists of two main components:
        1. A multi-head self-attention block.
        2. A feed-forward network (position-wise fully connected layer).
        
        Each of these components is wrapped in a residual connection with layer normalization and dropout.

        Args:
            self_attention_block (MultiHeadAttentionBlock): The multi-head self-attention mechanism.
            feed_forward_block (FeedForwardBlock): The feed-forward network.
            dropout (float): Dropout rate to be applied in the residual connections.

        Attributes:
            self_attention_block (MultiHeadAttentionBlock): Handles the self-attention for the block.
            feed_forward_block (FeedForwardBlock): Processes the output of the self-attention.
            residual_connections (nn.ModuleList): Contains two residual connections for the self-attention and feed-forward blocks.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # Two residual connections: one for self-attention and one for the feed-forward block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        """
        Forward pass through the encoder block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            src_mask (Tensor or None): Optional source mask to block certain positions in the attention mechanism.

        Returns:
            Tensor: The output of the encoder block, which has passed through self-attention and the feed-forward network.
        """
        # Apply the first residual connection for the self-attention block
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # Apply the second residual connection for the feed-forward block
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    """
    Describes the role of the encoder, which stacks multiple EncoderBlock modules.
    The forward method includes detailed comments on how the input passes through each encoder layer
    before being normalized.
    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        """
        The encoder module in the Transformer.

        The encoder consists of a stack of identical encoder blocks, each containing a multi-head
        self-attention mechanism and a feed-forward network.

        Args:
            layers (nn.ModuleList): A list of `EncoderBlock` modules.

        Attributes:
            layers (nn.ModuleList): A stack of encoder layers (blocks).
            norm (LayerNormalization): Applies layer normalization to the final output of the encoder.
        """
        super().__init__()
        self.layers = layers  # A list of encoder blocks
        self.norm = LayerNormalization(features)  # Apply layer normalization at the end of the encoder

    def forward(self, x, mask):
        """
        Forward pass through the encoder.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor or None): Optional mask to prevent attention to certain positions.

        Returns:
            Tensor: The final output of the encoder, after passing through all layers and normalization.
        """
        # Pass the input through each encoder block (layer)
        for layer in self.layers:
            x = layer(x, mask)
        # Apply layer normalization to the final output
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(self,
                 features: int,
                 self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float
    ) -> None:
        """
        Represents a single block in the decoder of the Transformer.

        Each decoder block consists of three main components:
        1. A masked multi-head self-attention block.
        2. A cross-attention block (which attends to the encoder's output).
        3. A feed-forward network (position-wise fully connected layer).
        
        Each of these components is wrapped in a residual connection with layer normalization and dropout.

        Args:
            self_attention_block (MultiHeadAttentionBlock): The masked multi-head self-attention mechanism.
            cross_attention_block (MultiHeadAttentionBlock): The multi-head attention block for encoder-decoder attention.
            feed_forward_block (FeedForwardBlock): The feed-forward network.
            dropout (float): Dropout rate applied in the residual connections.

        Attributes:
            self_attention_block (MultiHeadAttentionBlock): Handles self-attention for the decoder.
            cross_attention_block (MultiHeadAttentionBlock): Performs encoder-decoder cross-attention.
            feed_forward_block (FeedForwardBlock): Processes the output of the cross-attention.
            residual_connections (nn.ModuleList): Contains three residual connections for the self-attention, cross-attention, and feed-forward blocks.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        # Three residual connections: for self-attention, cross-attention, and feed-forward block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass through the decoder block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, tgt_seq_len, d_model), where `tgt_seq_len` is the length of the target sequence.
            encoder_output (Tensor): Output from the encoder of shape (batch_size, src_seq_len, d_model), where `src_seq_len` is the length of the source sequence.
            src_mask (Tensor or None): Source mask to prevent attending to certain positions in the source sequence.
            tgt_mask (Tensor or None): Target mask to prevent attending to future positions in the target sequence (for autoregressive decoding).

        Returns:
            Tensor: The output of the decoder block after passing through self-attention, cross-attention, and the feed-forward network.
        """
        # 1. Self-attention with masking (to avoid looking at future tokens)
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # 2. Cross-attention: the decoder attends to the encoder's output
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        # 3. Feed-forward block to further process the output
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        """
        The decoder module in the Transformer.

        The decoder consists of a stack of identical decoder blocks, each containing:
        1. A masked multi-head self-attention mechanism.
        2. A cross-attention block (attending to the encoder output).
        3. A feed-forward network (position-wise fully connected layer).
        
        Args:
            layers (nn.ModuleList): A list of `DecoderBlock` modules.

        Attributes:
            layers (nn.ModuleList): A stack of decoder layers (blocks).
            norm (LayerNormalization): Applies layer normalization to the final output of the decoder.
        """

        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass through the decoder.

        Args:
            x (Tensor): Input tensor of shape (batch_size, tgt_seq_len, d_model), where `tgt_seq_len` is the length of the target sequence.
            encoder_output (Tensor): Output from the encoder of shape (batch_size, src_seq_len, d_model), where `src_seq_len` is the length of the source sequence.
            src_mask (Tensor or None): Source mask to prevent attending to certain positions in the source sequence.
            tgt_mask (Tensor or None): Target mask to prevent attending to future positions in the target sequence (for autoregressive decoding).

        Returns:
            Tensor
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)



class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        The final layer of the Transformer model that projects the output of the decoder
        into the vocabulary space, converting the model's output to predicted token probabilities.

        Args:
            d_model (int): The dimensionality of the decoder's output (must match the model's internal representation size).
            vocab_size (int): The number of unique tokens in the output vocabulary.

        Attributes:
            proj (nn.Linear): A linear layer that maps from the model's hidden dimension (`d_model`) to the vocabulary size (`vocab_size`).
        """
        super().__init__()
        # Linear layer maps from (d_model) --> (vocab_size)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        Forward pass through the projection layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model), where `seq_len` is the length of the output sequence.

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, vocab_size), representing the predicted probabilities for each token in the vocabulary.
        """
        # Apply the linear transformation followed by log softmax to get token probabilities
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)


class Transformer(nn.Module):

    def __init__(self, 
                 encoder: Encoder,
                 decoder: Decoder, 
                 src_embed: InputEmbeddings, 
                 tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer,
    ) -> None:
        """
        The Transformer model that consists of an encoder-decoder architecture. It takes a source input sequence,
        processes it through the encoder, and then the decoder generates an output sequence based on the encoded input.

        Args:
            encoder (Encoder): The encoder component of the Transformer, which processes the source input.
            decoder (Decoder): The decoder component of the Transformer, which generates the target output sequence.
            src_embed (InputEmbeddings): The embedding layer for the source input sequence.
            tgt_embed (InputEmbeddings): The embedding layer for the target output sequence.
            src_pos (PositionalEncoding): The positional encoding layer for the source sequence, which adds positional information to the embeddings.
            tgt_pos (PositionalEncoding): The positional encoding layer for the target sequence, which adds positional information to the embeddings.
            projection_layer (ProjectionLayer): The final projection layer that converts the decoder's output to token probabilities in the vocabulary space.

        Attributes:
            encoder (Encoder): Encodes the input sequence.
            decoder (Decoder): Decodes the encoded input to generate an output sequence.
            src_embed (InputEmbeddings): Embeds the source input sequence.
            tgt_embed (InputEmbeddings): Embeds the target sequence.
            src_pos (PositionalEncoding): Adds positional encodings to the source input embeddings.
            tgt_pos (PositionalEncoding): Adds positional encodings to the target input embeddings.
            projection_layer (ProjectionLayer): Projects the output of the decoder to the vocabulary space for token prediction.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.decoder = decoder
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        """
        Encodes the source input sequence by first embedding the tokens and then applying positional encodings.

        Args:
            src (Tensor): Source input tensor of shape (batch_size, src_seq_len), where `src_seq_len` is the length of the source sequence.
            src_mask (Tensor or None): Mask for the source sequence, used to prevent the model from attending to padding tokens.

        Returns:
            Tensor: Encoded source sequence of shape (batch_size, src_seq_len, d_model).
        """
        # Apply input embeddings to the source sequence
        src = self.src_embed(src)
        # Add positional encodings to the embedded input
        src = self.src_pos(src)
        # Pass the embedded and encoded input through the encoder
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        """
        Decodes the target sequence using the encoded source representation and the target input embeddings.

        Args:
            encoder_output (Tensor): The encoded representation of the source sequence, of shape (batch_size, src_seq_len, d_model).
            src_mask (Tensor or None): Mask for the source sequence to prevent attention to certain tokens (like padding).
            tgt (Tensor): Target sequence tensor of shape (batch_size, tgt_seq_len), where `tgt_seq_len` is the length of the target sequence.
            tgt_mask (Tensor or None): Mask for the target sequence to prevent the model from attending to future tokens during decoding.

        Returns:
            Tensor: Decoded target sequence of shape (batch_size, tgt_seq_len, d_model).
        """
        # Embed the target input sequence
        tgt = self.tgt_embed(tgt)
        # Add positional encodings to the target embeddings
        tgt = self.tgt_pos(tgt)
        # Pass through the decoder using the encoded source representation
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        """
        Projects the decoder's output into the vocabulary space to predict the next token.

        Args:
            x (Tensor): The decoder output tensor of shape (batch_size, tgt_seq_len, d_model).

        Returns:
            Tensor: The projected output of shape (batch_size, tgt_seq_len, vocab_size), representing token probabilities for the target sequence.
        """
        # Pass the decoder's output through the projection layer
        return self.projection_layer(x) 



def build_transformer(src_vocab_size: int, 
                      tgt_vocab_size: int, 
                      src_seq_len: int, 
                      tgt_seq_len: int, 
                      d_model: int = 512,
                      N: int = 6,
                      h: int = 8,
                      dropout: float = 0.1,
                      d_ff: int = 2048,
) -> Transformer:
    """
    Builds a Transformer model by creating the necessary components such as embeddings, 
    positional encodings, encoder and decoder blocks, and projection layers.

    Args:
        src_vocab_size (int): The size of the source vocabulary (number of unique tokens in the source language).
        tgt_vocab_size (int): The size of the target vocabulary (number of unique tokens in the target language).
        src_seq_len (int): Maximum sequence length for the source input.
        tgt_seq_len (int): Maximum sequence length for the target input.
        d_model (int, optional): Dimensionality of the model's internal representation. Default is 512.
        N (int, optional): The number of encoder and decoder blocks. Default is 6.
        h (int, optional): The number of attention heads in each multi-head attention block. Default is 8.
        dropout (float, optional): Dropout rate to prevent overfitting. Default is 0.1.
        d_ff (int, optional): Dimensionality of the feed-forward network in each block. Default is 2048.

    Returns:
        Transformer: A fully constructed Transformer model ready for training.

    Components:
        - Input Embeddings: Converts token indices into dense vectors.
        - Positional Encoding: Adds positional information to the embeddings.
        - Encoder Blocks: N layers consisting of multi-head self-attention and feed-forward networks.
        - Decoder Blocks: N layers consisting of self-attention, cross-attention, and feed-forward networks.
        - Projection Layer: Projects the decoder output into the vocabulary space for generating predictions.

    The model is initialized using Xavier uniform initialization for the parameters.
    """
    # Create the embedding layers for source and target vocabularies
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers to add positional information to embeddings
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        # Multi-head attention block for self-attention in the encoder
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        # Feed-forward block in the encoder
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        # Combine the blocks into an encoder block
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    # Created the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        # Multi-head attention for self-attention in the decoder
        decode_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        # Multi-head attention for cross-attention (encoder-decoder attention)
        decode_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        # Feed-forward block in the decoder
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        # Combine the blocks into a decoder block
        decoder_block = DecoderBlock(d_model, decode_self_attention_block, decode_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and the decoder by stacking their respective blocks
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer to map the decoder's output to the target vocabulary space
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the Transformer model by combining all the components
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters using Xavier uniform initialization
    for p in transformer.parameters():
        if p.dim() > 1:  # Only apply to weights (not biases) 
            nn.init.xavier_uniform_(p)

    return transformer