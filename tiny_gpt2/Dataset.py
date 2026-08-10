from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride=1):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i+1:i + max_length + 1]
            self.input_ids.append(input_chunk)
            self.target_ids.append(target_chunk)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_data_loader(text, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

if __name__ == "__main__":
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("--- test dataset ---")
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(raw_text, tokenizer, max_length=4, stride=1)
    print(dataset[0])

    print("--- test dataloader ---")
    dataloader = create_data_loader(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    second_batch = next(data_iter)
    print(first_batch)
    print(second_batch)

    print("--- test dataloader ---")
    dataloader = create_data_loader(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print(inputs)
    print(targets)

