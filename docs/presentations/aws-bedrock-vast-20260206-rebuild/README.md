# AWS Bedrock x VAST DATA Rebuild Deck

- Source deck: `C:\Users\unnow\OneDrive - 株式会社ネットワールド\共有ドキュメント\2025.12.26_AWS Bedrock & VAST DATA 検証\AWS Bedrock & VAST DATA検証プラン_20260206.pptx`
- Template: `C:\pptx-creator\template\Networld-Basic.pptx`
- Rebuilt output: `output/AWS_Bedrock_VAST_restructured_20260302.pptx`

## Build

```powershell
pwsh .\scripts\build.ps1
```

## Files

- `context.json`: reconstruction context and story settings
- `slide-spec.json`: 12-slide compressed story (cover included in rendered deck)
- `scripts/render_networld_pptx.py`: template renderer
- `scripts/build.ps1`: one-command build entrypoint
