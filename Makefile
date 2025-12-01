# Makeの設定（デフォルトをrunにする）
.PHONY: all setup run build rebuild push

all: run

IMAGE_NAME=wadajun8/omnivla-img:v1

# ==========================================================
# 1. セットアップ (初回のみ: make setup)
# ==========================================================
setup:
	@echo "Installing Git LFS..."
	# エラーが出る場合は 'sudo apt install git-lfs' を実行してください
	git lfs install
	
	@echo "Downloading model checkpoints..."
	# フォルダが存在してもエラーにならないように || true をつけています
	git lfs clone https://huggingface.co/NHirose/omnivla-original || true
	git lfs clone https://huggingface.co/NHirose/omnivla-original-balance || true
	git lfs clone https://huggingface.co/NHirose/omnivla-finetuned-cast || true
	@echo "Setup Done!"

# ==========================================================
# 2. 実行 (普段使い: make run または make)
# ==========================================================
run:
	docker run --gpus all -it --rm -v $(CURDIR):/app $(IMAGE_NAME)

# ==========================================================
# 3. 管理用
# ==========================================================
build:
	docker build -t $(IMAGE_NAME) .

rebuild:
	docker build --no-cache -t $(IMAGE_NAME) .

push:
	docker push $(IMAGE_NAME)
