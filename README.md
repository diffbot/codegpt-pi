# @diffbot/codegpt-pi

Bring your existing [CodeGPT](https://www.codegpt.co/) free or paid subscription to Pi. This extension registers a custom CodeGPT provider to your Pi harness, allowing you to authenticate your CodeGPT subscription and use free, economy, and premium models in Pi.

## Install

```bash
# from npm
pi install npm:@diffbot/codegpt-pi

# from git
pi install git:github.com/diffbot/codegpt-pi
```

## Authentication

Run `/login` and pick **CodeGPT**. Follow the instructions to finish authentication.

## Models

Select an available CodeGPT model in Pi by running `/models`. Models are marked by their tier (free, economy, premium). Keep in mind that model usage in Pi may consume credits depending on your CodeGPT plan.

## Env overrides

| Variable | Purpose |
|---|---|
| `CODEGPT_SESSION_TOKEN` | Session token, skips the browser flow |
| `CODEGPT_ORG_ID` | Organization id header |
| `CODEGPT_BACKEND_URL` | API base (default `https://api.codegpt.co/api/v1`) |
| `CODEGPT_PLAYGROUND_URL` | Login base (default `https://app.codegpt.co`) |

## Develop

```bash
git clone https://github.com/diffbot/codegpt-pi.git
cd codegpt-pi
npm install
npm run typecheck

pi -e .
```

## Package layout

Follows the [Pi package](https://pi.dev/docs/latest/packages#creating-a-pi-package) conventions:

```
codegpt-pi/
├── extensions/     # the codegpt provider
├── assets/         # gallery preview
├── package.json    # pi manifest + pi-package keyword
└── README.md
```

## License

MIT
