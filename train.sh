#!/bin/bash
export USER_ID="$(id -u)"
export GROUP_ID="$(id -g)"
docker compose --project-directory docker/ up --build -d