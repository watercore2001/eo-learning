from geosprite.learning.model.time_series.encoder import Transformer, TransformerArgs
from geosprite.learning.model.image.encoder import WindowTransformer, SwinTransformerStages, SwinTransformerStagesArgs
from geosprite.learning.model.image.decoder import SegformerDecoder, ReshapeDecoder
from geosprite.learning.model.video.encoder import TemporalSpaceEncoder
from geosprite.learning.model.image.header import ReshapeHeader
from geosprite.learning.lightning_module import ClassificationModule
import torch
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis, flop_count_table


def output(model, x):
    flops = FlopCountAnalysis(model, x)
    print(flop_count_table(flops))


def transformer_info():
    x = torch.randn((64 * 64, 50, 128))
    model = Transformer(
        TransformerArgs(embedding_dim=128, depth=4, heads=4, head_dim=32, mlp_ratio=2, dropout=0))

    flops = FlopCountAnalysis(model, x)
    print(flops.total())


def window_transformer_info():
    model = WindowTransformer(window_size=64, use_absolute_position_embedding=True,
                              transformer_args=TransformerArgs(
                                  embedding_dim=128, depth=4, heads=4, head_dim=32, mlp_ratio=2, dropout=0))

    x = torch.randn((1, 128, 64, 64))
    output(model, x)


def swin_transformer_info():
    embedding_dim = 128
    depth_in_stages = [4]
    heads_in_stage = [128 * 2 ** i // 32 for i in range(len(depth_in_stages))]
    out_indices = [i for i in range(len(depth_in_stages))]
    model = SwinTransformerStages(stages_args=SwinTransformerStagesArgs(use_absolute_position_embedding=True,
                                                                        use_relative_position_embedding=False,
                                                                        embedding_dim=embedding_dim,
                                                                        depth_in_stages=depth_in_stages,
                                                                        heads_in_stages=heads_in_stage,
                                                                        out_indices=out_indices, window_size=16,
                                                                        mlp_ratio=2, dropout=0))
    x = torch.randn((20, 128, 64, 64))
    output(model, x)


def segformer_decoder_info():
    input_dim = 128
    stages = 2
    out_dim = 128
    dims_in_stages = [input_dim * 2 ** i for i in range(stages)]
    model = SegformerDecoder(dims_in_stages=dims_in_stages, out_dim=out_dim)
    x1 = torch.rand((20, input_dim, 64, 64))
    x2 = torch.rand((20, input_dim * 2, 32, 32))
    x3 = torch.rand((20, input_dim * 4, 16, 16))
    x4 = torch.rand((20, input_dim * 8, 8, 8))
    summary(model, input_data=[(x1, x2)], device="cpu")


def reshape_decoder_info():
    input_dim = 128
    stages = 2
    out_dim = 64
    dims_in_stages = [input_dim * 2 ** i for i in range(stages)]
    model = ReshapeDecoder(dims_in_stages=dims_in_stages, out_dim=out_dim)
    x1 = torch.rand((20, input_dim, 64, 64))
    x2 = torch.rand((20, input_dim * 2, 32, 32))
    x3 = torch.rand((20, input_dim * 4, 16, 16))
    x4 = torch.rand((20, input_dim * 8, 8, 8))
    summary(model, input_data=[(x1, x2)], device="cpu")


def reshape_header_info():
    model = ReshapeHeader(num_classes=20, embedding_dim=128, patch_size=2)
    x = torch.randn((20, 128, 64, 64))
    flops = FlopCountAnalysis(model, x)
    print(flop_count_table(flops))


def tsvit():
    temporal_encoder = Transformer(
        TransformerArgs(embedding_dim=128, depth=4, heads=4, head_dim=32, mlp_ratio=2, dropout=0))
    space_encoder = SwinTransformerStages(stages_args=SwinTransformerStagesArgs(use_absolute_position_embedding=True,
                                                                                use_relative_position_embedding=False,
                                                                                embedding_dim=128,
                                                                                depth_in_stages=[4],
                                                                                heads_in_stages=[4],
                                                                                out_indices=[0],
                                                                                window_size=16,
                                                                                mlp_ratio=2,
                                                                                dropout=0))

    encoder = TemporalSpaceEncoder(image_channels=10, patch_size=2, embedding_dim=128, num_classes=20,
                                   temporal_encoder=temporal_encoder, space_encoder=space_encoder)
    header = ReshapeHeader(num_classes=20, embedding_dim=128, patch_size=2)
    model = ClassificationModule(encoder=encoder, header=header)
    x = torch.randn((1, 20, 10, 64, 64))
    dates = torch.randint(low=1, high=300, size=(1, 20))

    output(model, ([x, dates]))


if __name__ == "__main__":
    window_transformer_info()
    # swin_transformer_info()
    # segformer_decoder_info()
    # reshape_header_info()
    # reshape_decoder_info()
    # tsvit()
