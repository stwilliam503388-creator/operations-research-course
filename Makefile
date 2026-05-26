# 运筹学与优化方法课程系列 — 一键验证
# Operations Research & Optimization Course Series — One-Click Verification
# 用法 / Usage:
#   make          → 列出所有代码 / List all code
#   make python   → 运行所有 Python 代码验证 / Run all Python code
#   make list     → 列出所有可运行代码 / List all runnable code
#   make clean    → 清理缓存文件 / Clean cache files

.PHONY: all python list clean

PYTHON := python3

# 查找所有 Python 代码（排除 venv、cache 和 prompts 目录）
# Find all Python files (exclude venv, cache, and prompts directories)
PY_FILES := $(shell find . -name '*.py' -not -path '*/venv/*' -not -path '*/.venv/*' -not -path '*/__pycache__/*' -not -path '*/prompts/*' | sort)

all: list

list:
	@echo "========================================"
	@echo "  运筹学与优化方法课程系列 — 代码清单"
	@echo "  OR & Optimization Course Series — Code List"
	@echo "========================================"
	@echo ""
	@echo "Python 代码 ($(words $(PY_FILES)) 个):"
	@for f in $(PY_FILES); do \
		printf "  %s\n" "$$f"; \
	done
	@echo ""
	@echo "用法 / Usage:"
	@echo "  make python  — 运行所有 Python 代码（验证）/ Run all Python (verify)"
	@echo "  make clean   — 清理缓存文件 / Clean cache files"

python:
	@echo "========================================"
	@echo "  运行所有 Python 代码 / Running all Python"
	@echo "========================================"
	@errors=0; count=0; \
	for f in $(PY_FILES); do \
		echo "--- $$f ---"; \
		if $(PYTHON) "$$f" > /tmp/$$(basename $$f).log 2>&1; then \
			echo "  ✅"; \
		else \
			head -3 /tmp/$$(basename $$f).log; \
			echo "  ❌"; \
			errors=$$((errors+1)); \
		fi; \
		count=$$((count+1)); \
	done; \
	echo ""; \
	echo "$$count 个运行完成 / completed, $$errors 个失败 / failed"

clean:
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	@echo "已清理缓存文件 / Cache files cleaned"
