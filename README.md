# VintedAutoPrepare

A Windows desktop app that turns folders of item photos into ready-to-post **Vinted** listings.

For each item it:

1. **Fixes the photos**: rotates sideways phone shots, straightens slight tilts, gently corrects
   brightness/contrast/white balance, finds the item and **crops it centred at 4:5**
   (the ratio Vinted uses in its feed). The background is kept.
2. **Reads the item** with a free vision AI: brand, size, colour and **material/fabric**
   (from the care label when visible), or the equivalent for shoes, homeware, toys, books…
3. **Writes the listing**: a title (≤ 60 characters) and a short, friendly description with
   hashtags, in the language you pick (Greek, English, French, Italian, …).
4. **Proposes a price** in EUR for the Greek Vinted market, with a one-line explanation.

You review everything, edit if needed, and paste it into Vinted with the **Copy** buttons.

---

## 1. Install

**Option A – download the .exe**
Every push builds `VintedAutoPrepare.exe` on GitHub: open the repository's **Actions** tab →
latest *Windows build* run → download the `VintedAutoPrepare-windows` artifact and unzip it.
Windows SmartScreen may warn because the exe isn't signed: *More info → Run anyway*.

**Option B – build it yourself**
Install [Python 3.11+](https://www.python.org/downloads/) (tick *Add python.exe to PATH*),
then double-click `build_windows.bat`. The result is `dist\VintedAutoPrepare.exe`.

**Option C – run from source (two double-clicks)**
Install [Python 3.11+](https://www.python.org/downloads/) (tick *Add python.exe to PATH*), then in the project folder:

1. Double-click **`Install.bat`**. It creates the `venv` folder, updates pip, installs the
   requirements and creates a **`VintedAutoPrepare`** shortcut in the project folder.
   Run it again at any time to update everything (or if you move the folder).
2. Double-click the **`VintedAutoPrepare`** shortcut to open the program. You can copy it to
   your Desktop or pin it to the Start menu.

> The first time photos are processed, the app downloads its item-detection model
> (~170 MB, once) to `%APPDATA%\VintedAutoPrepare\models`. If that's not possible, it
> falls back to a simpler detector automatically.

## 2. Choose your free AI (Settings ⚙)

| | Google Gemini (default) | Ollama (offline) |
|---|---|---|
| Cost | Free tier | Free |
| Quality | Very good (reads small label text well) | Good |
| Needs | Internet + free API key | Ollama + a vision model, a GPU with ≥ 8 GB is ideal |
| Privacy | Photos go to Google; free-tier data may be used to improve Google's models | Nothing leaves your PC |

**Gemini:** go to <https://aistudio.google.com/apikey>, sign in with a Google account,
*Create API key*, and paste it into Settings → **API key**. Press **Refresh** to check it
works and to list models. The default `gemini-flash-latest` always points to Google's current
Flash model. The free tier has per-minute and per-day limits, which are plenty for a few dozen
items a day. If you hit a limit, wait and press *Rewrite listing*.

**Ollama:** install from <https://ollama.com/download>, then in a terminal:

```bat
ollama pull qwen2.5vl:7b
```

In Settings choose **Ollama**, press **Refresh**, and pick the model. Any Ollama *vision*
model works (e.g. `gemma3:12b`, or newer Qwen-VL releases). Larger models read labels
better but are slower.

You can switch between the two at any time.

## 3. Use it

1. Put each item's photos in **its own folder** (front, back, details, **care label**, size
   label…). Folder name = item name.
2. Drag one or more item folders onto the window, or drag a parent folder that contains
   item folders. You can also use **Add folder…**.
3. Optional: type notes for an item, e.g. *"worn twice"*, *"small mark on left sleeve"*,
   *"runs small"*. The AI treats them as facts. The item's **condition is only mentioned if
   you write it in the notes**.
4. Pick the **listing language** in the toolbar and press **▶ Process all**.
5. Click an item to review it: edit anything, then use **Copy** next to each field.
   The description's Copy button includes the hashtags.

Tips:

- Right-click a photo → **Use full photo (no crop)** if an auto-crop isn't right.
- **Rewrite listing** asks the AI again (e.g. after adding notes) without reprocessing the photos.
- Items you processed before reload instantly when you add the same folder again.
- Clear care-label and size-label close-ups make brand, size and material far more accurate.

## 4. Output

Results go to `Documents\VintedAutoPrepare\<item folder name>\` (changeable in Settings):

```
01.jpg 02.jpg …   processed photos, original order, location data (EXIF) removed
listing.txt       title, description + hashtags, details, price, ready to paste
listing.json      everything, used to reopen the item later
```

Your original photos are never modified.

## Settings

| Setting | Default | |
|---|---|---|
| Photos sent to AI | 6 | More = better label reading, slower. Photos are picked evenly and always include the first and last. |
| Straighten | on, up to 5° | Uses straight lines in the scene (floorboards, door frames). |
| Auto enhance | on, 60 % | Brightness, contrast and a careful white balance that won't grey out colourful items. |
| Space around item | 8 % | Margin kept around the item when cropping. |

Settings live in `%APPDATA%\VintedAutoPrepare\settings.json`, and a log in `app.log`
next to it. The Gemini key is stored there in plain text, or you can set the
`GEMINI_API_KEY` environment variable instead.

## Development

```bash
pip install -r requirements-dev.txt
QT_QPA_PLATFORM=offscreen python -m pytest      # all tests, including a UI smoke test
VINTED_MODEL_DIR=/path/to/models python -m pytest   # also test the real U^2-Net detector
```

Code layout:

```
vinted_prep/
  imaging/   io (load/EXIF/save), straighten, enhance, detect (U^2-Net + OpenCV fallback), crop, pipeline
  ai/        prompt + JSON parsing, gemini, ollama
  processor.py   one item end-to-end (UI-independent)
  export.py      output folder, listing.txt/json, reload
  ui/            PySide6 main window, item panel, settings, background jobs
```

The prices are AI estimates, not market data, so check them against similar listings for
anything valuable. Always double-check brand and size before posting.
