# New project with template

<!-- git remote set-url origin git@github.com:titusfx/transcription-with-face-source.git -->

git remote set-url origin <NEW-project>

<!-- replace `scoreboard-padel` with `face_transcriber` (better to use _ than -) -->

replace `scoreboard-padel` with `project_name` (better to use \_ than -)

<!-- replace `scoreboard_padel` with `face_transcriber` (better to use _ than -) -->

replace `scoreboard_padel` with `project_name` (better to use \_ than -)

# Start

poetry install
poetry run gradio
poetry run client-gradio

# Start with docker

```bash
docker compose up
```

# Config

Configuration Is to avoid adding in .env in the repo and share the file.
In this case PORT should be in .env so we can update easy docker compose.yml, client and server. But I will require to share .env
