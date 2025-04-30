# 4_run_model_without_cuda

.gguf 形式に変換したモデルを実行します。
3_convert_unsolth_to_gguf では、.gguf 形式に変換する際に量子化を行なっており、そのモデルを利用した場合は CUDA を搭載していない PC や IoT 機器でも LLM モデルを実行できます。
