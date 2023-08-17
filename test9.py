
def add():
    import torch
    from vector_quantize_pytorch import VectorQuantize

    vq = VectorQuantize(
        dim = 256,
        codebook_size = 512,     # codebook size
        decay = 0.8,             # the exponential moving average decay, lower means the dictionary will change faster
        commitment_weight = 1.   # the weight on the commitment loss
    )

    x = torch.randn(1, 1024, 256)
    quantized, indices, commit_loss = vq(x) # (1, 1024, 256), (1, 1024), (1)

    from IPython import embed; embed(using=False); os._exit(0)


def add2():
    import torch
    from vector_quantize_pytorch import ResidualVQ

    residual_vq = ResidualVQ(
        dim = 256,
        num_quantizers = 8,      # specify number of quantizers
        codebook_size = 1024,    # codebook size
    )

    x = torch.randn(1, 1024, 256)

    quantized, indices, commit_loss = residual_vq(x)

    # (1, 1024, 256), (1, 1024, 8), (1, 8)
    # (batch, seq, dim), (batch, seq, quantizer), (batch, quantizer)

    # if you need all the codes across the quantization layers, just pass return_all_codes = True

    quantized, indices, commit_loss, all_codes = residual_vq(x, return_all_codes = True)

    # *_, (8, 1, 1024, 256)
    # all_codes - (quantizer, batch, seq, dim)

    from IPython import embed; embed(using=False); os._exit(0)


if __name__ == '__main__':
    add()