# Deploying to Hugging Face Spaces

Hugging Face Spaces can host this demo without buying a domain. The Space will get a public URL like:

```text
https://huggingface.co/spaces/LittleOtterUD/INDENGInterface
```

## 1. Create the deploy bundle

From the project root:

```bash
./interactive_demo/export_hf_space.sh
```

This creates:

```text
scratch/huggingface_space/
  README.md
  Dockerfile
  requirements.txt
  app.py
  static/
  src/food_project/
  model/best_model.pt
```

The checkpoint is about 128 MB, so it should be uploaded as a large file.

## 2. Create the Space

The current Space is:

```text
LittleOtterUD/INDENGInterface
```

It should be configured with `Docker` as the SDK. If it was created with another SDK, open the Space settings and switch the SDK to Docker, or recreate the Space with Docker selected.

## 3. Upload with the Hugging Face CLI

Install and log in:

```bash
pip install -U huggingface_hub
hf auth login
```

Upload the bundle:

```bash
hf upload LittleOtterUD/INDENGInterface scratch/huggingface_space . --repo-type=space
```

## Alternative: Git + Git LFS

If you prefer git:

```bash
git clone https://huggingface.co/spaces/LittleOtterUD/INDENGInterface
cp -R scratch/huggingface_space/. INDENGInterface/
cd INDENGInterface
git lfs install
git lfs track "model/*.pt"
git add .
git commit -m "Deploy food health classifier"
git push
```

If `git lfs` is missing on macOS:

```bash
brew install git-lfs
```

## Notes

- The Docker container serves the app on `0.0.0.0:7860`, which matches the Space `app_port`.
- The deployed demo uses CPU by default. That is fine for single-image ResNet18 inference, just a little slower than local MPS.
- If the Space build fails because a package version is unavailable, use the same versions from the project environment or relax the exact pins in `requirements.txt`.

## Troubleshooting 403 Upload Errors

If upload fails with a message like:

```text
403 Forbidden: Authorization error
Cannot access ... /info/lfs/objects/batch
```

the active Hugging Face account is probably correct, but the token does not have write permission for this Space's large files.

Fastest fix:

1. Open `https://huggingface.co/settings/tokens`.
2. Create a new token with role `Write`.
3. Run `hf auth login` again and paste the new token.
4. Choose `y` for "Add token as git credential?"
5. Retry:

```bash
hf upload LittleOtterUD/INDENGInterface scratch/huggingface_space . --repo-type=space --commit-message "Deploy interactive food classifier"
```

More restrictive fix:

Create a fine-grained token that explicitly includes the Space `LittleOtterUD/I242BProject` and grants write access to repository contents.
