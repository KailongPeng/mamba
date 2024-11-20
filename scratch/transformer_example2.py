import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k  # 使用Multi30k数据集作为例子


# 数据预处理函数
def yield_tokens(data_iter, text_field):
    """从数据迭代器中生成token序列"""
    for _, example in data_iter:
        yield [text_field.init_token] + text_field.tokenize(example.src) + [text_field.eos_token]


# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, ntoken, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer

        # 创建位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # 创建嵌入层
        self.encoder = nn.Embedding(ntoken, d_model)
        self.decoder = nn.Embedding(ntoken, d_model)
        # 创建Transformer的编码器和解码器层
        self.transformer_encoder = TransformerEncoder(TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout),
                                                      num_encoder_layers)
        self.transformer_decoder = TransformerDecoder(TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout),
                                                      num_decoder_layers)
        # 创建输出层
        self.out = nn.Linear(d_model, ntoken)

    def forward(self, src, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask,
                memory_key_padding_mask):
        # 前向传播函数
        src = self.encoder(src) * math.sqrt(self.d_model)  # 缩放嵌入向量
        src = self.pos_encoder(src)  # 添加位置编码
        memory = self.transformer_encoder(src, src_mask, src_key_padding_mask)  # 编码器输出
        tgt = self.decoder(tgt) * math.sqrt(self.d_model)  # 目标序列的嵌入
        tgt = self.pos_encoder(tgt)  # 添加位置编码
        output = self.transformer_decoder(tgt, memory, tgt_mask, None, tgt_key_padding_mask,
                                          memory_key_padding_mask)  # 解码器输出
        output = self.out(output)  # 输出层
        return output


# 定义位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# 加载数据
train_iter, valid_iter, test_iter = Multi30k(split=('train', 'valid', 'test'))
tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer))

# 模型参数设置
ntokens = len(vocab)  # 单词表大小
emsize = 512  # 嵌入维度
nhid = 2048  # 前馈层维度
nlayers = 6  # 编码器/解码器层数
nhead = 8  # 多头注意力机制中的头数
dropout = 0.5  # Dropout比率
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备选择

# 初始化模型
model = TransformerModel(ntokens, emsize, nhead, nlayers, nlayers, nhid, dropout).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练模型
def train(model, dataloader, optimizer, criterion, device):
    model.train()  # 设置模型为训练模式
    total_loss = 0.
    for batch in dataloader:
        src, tgt = batch.src.to(device), batch.tgt.to(device)  # 获取源语言和目标语言数据
        src_mask, tgt_mask = generate_square_subsequent_masks(src.size(0), tgt.size(0)).to(device)
        optimizer.zero_grad()  # 清空梯度
        output = model(src, tgt, src_mask, tgt_mask, None, None, None)  # 前向传播
        loss = criterion(output.view(-1, ntokens), tgt.view(-1))  # 计算损失
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数
        total_loss += loss.item()
    return total_loss / len(dataloader)


# 测试模型
def evaluate(model, dataloader, criterion, device):
    model.eval()  # 设置模型为评估模式
    total_loss = 0.
    with torch.no_grad():
        for batch in dataloader:
            src, tgt = batch.src.to(device), batch.tgt.to(device)  # 获取源语言和目标语言数据
            src_mask, tgt_mask = generate_square_subsequent_masks(src.size(0), tgt.size(0)).to(device)
            output = model(src, tgt, src_mask, tgt_mask, None, None, None)  # 前向传播
            loss = criterion(output.view(-1, ntokens), tgt.view(-1))  # 计算损失
            total_loss += loss.item()
    return total_loss / len(dataloader)


# 主程序
epochs = 10  # 训练轮数
for epoch in range(1, epochs + 1):
    train_loss = train(model, train_iter, optimizer, criterion, device)
    val_loss = evaluate(model, valid_iter, criterion, device)
    print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# 测试模型性能
test_loss = evaluate(model, test_iter, criterion, device)
print(f'Test Loss: {test_loss:.4f}')