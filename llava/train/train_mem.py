from llava.train.train import train, test

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
    # test(attn_implementation="flash_attention_2")
