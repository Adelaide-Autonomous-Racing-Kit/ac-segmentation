.ONESHELL:
SHELL = /bin/zsh

setup-pre-push:
	@echo "Setting up pre-push hook..."
	@cp pre_push.sh .git/hooks/pre-push
	@chmod +x .git/hooks/pre-push
	@echo "Done!"