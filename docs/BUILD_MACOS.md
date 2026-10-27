# macOS (Apple Silicon) 빌드 가이드

M1/M2/M3 Mac에서 `PhotoDesk.app`(arm64)을 만드는 절차입니다.
같은 과정이 `.github/workflows/build.yml`의 `macos-14` 작업에도 들어 있습니다.

## 1. 준비물

| 항목 | 내용 |
|---|---|
| macOS | 11 Big Sur 이상, Apple Silicon |
| Python | 3.10 이상, **arm64 빌드**(python.org 설치본 또는 Homebrew `/opt/homebrew/bin/python3`) |
| Xcode 명령행 도구 | `xcode-select --install` (codesign, ditto 사용) |
| 원본 설치 파일 | `Photo_Desk_V_4_3_0.exe` (프로파일·UI·도움말 설치용, 앱에 포함하지 않음) |
| 테스트용 RAW | 선택 사항. `raw/*.DCR` |

Python이 arm64인지 먼저 확인합니다. `arm64`가 나와야 합니다. `x86_64`가 나오면 Rosetta용 Python이므로 다른 Python을 쓰세요.

```bash
python3 -c "import platform; print(platform.machine())"
```

## 2. 소스 받기와 가상 환경

```bash
git clone <저장소 주소> kodak-photodesk
cd kodak-photodesk
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

rawpy, numpy, opencv-python-headless, Pillow, tifffile, PySide6는 모두 arm64 휠이 있으므로 컴파일러 없이 설치됩니다.

## 3. 테스트

전체 테스트를 돌리려면 원본 설치 파일과 샘플 RAW를 아래 위치에 둡니다. 둘 다 git에 올리지 않습니다(`.gitignore`). 파일이 없으면 해당 테스트는 건너뜁니다.

```
vendor/original/Photo_Desk_V_4_3_0.exe
raw/4H6B9130.DCR
```

```bash
python -m pytest -q
```

GUI 점검은 프로파일을 설치한 뒤에 실행합니다(5절 참고).

```bash
python tests/gui_smoke.py raw /tmp/pd_gui.png
python tests/gui_commands.py raw /tmp/pdshots
```

두 스크립트는 `$TEMP`(없으면 `/tmp`)의 `photodesk_test.ini`를 설정 파일로 씁니다. 그래서 실제 사용자 설정은 바뀌지 않습니다.

## 4. Kodak 프로파일 설치

빌드하기 **전에** 원본 설치 파일에서 프로파일을 풀어 둡니다. 원본 설치 파일은 실행하지 않고 읽기만 하므로 Mac에서도 동작합니다.

```bash
python -m photodesk install-profiles vendor/original/Photo_Desk_V_4_3_0.exe
```

빌드할 때 spec 파일이 이 위치의 프로파일 57개와 번역표, 원본 화면 리소스 묶음(`PhotoDeskUI.pack`, 원본 `PhotoDsk.exe`에서 화면 리소스만 뽑은 것)을 `PhotoDesk.app/Contents/Resources/profiles`에 넣습니다.
원본 `PhotoDsk.exe`는 앱에 넣지 않습니다. 프로파일이 없으면 프로파일 없이 빌드되고, 앱을 처음 실행할 때 설치 파일을 묻습니다.

설치 위치는 `~/Library/Application Support/PhotoDesk/profiles`입니다. 프로파일 57개, `ui/PhotoDsk.exe`, `ui/PhotoDeskUIStrings.u16`이 들어갑니다.
GUI를 처음 실행할 때 프로파일이 없으면 설치 파일을 고르는 창이 뜨므로, 이 단계를 GUI에서 해도 됩니다.

다른 위치를 쓰려면 `PHOTODESK_PROFILES=/경로`를 지정합니다. UI 언어를 강제하려면 `PHOTODESK_LANG=Korean`을 지정합니다.

Custom Look 프로파일은 `/Library/ColorSync/Profiles` 또는 `~/Library/ColorSync/Profiles`에 두면 Look 메뉴에 나타납니다(원본 도움말 "Installing Profiles"와 같음).

## 5. 앱 빌드

```bash
pyinstaller --noconfirm packaging/photodesk.spec
```

결과물은 `dist/PhotoDesk.app`입니다. spec 파일이 macOS에서 `target_arch="arm64"`로 빌드하고 `BUNDLE`로 .app을 만듭니다. 번들 ID는 `org.photodesk.PhotoDesk`, 최소 OS는 11.0입니다.

빌드 확인:

```bash
file dist/PhotoDesk.app/Contents/MacOS/PhotoDesk        # "arm64"가 보여야 함
# 프로파일이 없으면 convert는 코드 2로 끝나야 함(충돌하지 않음)
dist/PhotoDesk.app/Contents/MacOS/PhotoDesk convert nonexistent.DCR -o out; echo $?
```

## 6. 실행 확인

```bash
open dist/PhotoDesk.app
# 폴더나 파일을 바로 열기
open -a dist/PhotoDesk.app --args ~/Pictures/DCS
dist/PhotoDesk.app/Contents/MacOS/PhotoDesk convert raw/4H6B9130.DCR -o /tmp/out -f tiff
```

## 7. 서명과 배포

### 직접 쓰거나 내부 배포 (애드혹 서명)

```bash
codesign --force --deep --sign - dist/PhotoDesk.app
ditto -c -k --keepParent dist/PhotoDesk.app PhotoDesk-macos-arm64.zip
```

인터넷으로 받은 zip을 풀면 Gatekeeper가 "확인되지 않은 개발자"라며 막습니다. 받은 쪽에서 한 번 아래를 실행하거나, Finder에서 앱을 우클릭 → 열기를 선택합니다.

```bash
xattr -dr com.apple.quarantine /Applications/PhotoDesk.app
```

### 정식 배포 (Developer ID + 공증)

Apple Developer 계정이 있을 때만 해당합니다.

```bash
codesign --force --deep --options runtime --timestamp \
  --sign "Developer ID Application: <이름> (<팀 ID>)" dist/PhotoDesk.app
ditto -c -k --keepParent dist/PhotoDesk.app PhotoDesk-macos-arm64.zip
xcrun notarytool submit PhotoDesk-macos-arm64.zip --apple-id <이메일> --team-id <팀 ID> \
  --password <앱 전용 암호> --wait
xcrun stapler staple dist/PhotoDesk.app
ditto -c -k --keepParent dist/PhotoDesk.app PhotoDesk-macos-arm64.zip   # 스테이플된 앱으로 다시 압축
```

## 8. GitHub Actions로 빌드

Mac이 없어도 저장소를 GitHub에 올리면 `build` 워크플로가 `macos-14`(Apple Silicon) 러너에서 테스트, 빌드, 애드혹 서명을 하고 `PhotoDesk-macos-arm64.zip`을 아티팩트로 올립니다. Actions 탭에서 수동 실행(`workflow_dispatch`)도 됩니다.

## 9. 알려진 macOS 차이점

- **단축키:** 메뉴 단축키는 원본 Windows 키를 그대로 씁니다. Qt가 macOS에서 `Ctrl`을 `⌘`로 바꿔 표시합니다.
- **외부 프로그램:** 이미지 편집기와 Camera Manager로 `.app`을 지정하면 `open -a`로 실행합니다.
- **검증 상태:** Windows에서는 실제 실행까지 확인했습니다. macOS 빌드는 GitHub Actions 러너에서 빌드·CLI 점검만 하도록 되어 있고, 실제 Mac에서 GUI를 확인한 적은 없습니다. 대화상자 글자 잘림 등이 보이면 알려 주세요.

## 문제 해결

| 증상 | 원인과 해결 |
|---|---|
| `file`에 `x86_64`가 나옴 | Rosetta Python으로 빌드함. arm64 Python으로 가상 환경을 다시 만듦 |
| `pip install`이 rawpy를 소스로 빌드하려 함 | Python 버전에 맞는 휠이 없음. Python 3.10–3.13 사용 |
| 앱이 바로 꺼짐 | 터미널에서 `dist/PhotoDesk.app/Contents/MacOS/PhotoDesk`를 직접 실행해 오류 확인 |
| "Kodak profiles not installed" | 4절의 `install-profiles` 실행 후 다시 빌드 |
| 앱이 "손상되었습니다"로 열리지 않음 | 격리 속성 문제. 7절의 `xattr -dr com.apple.quarantine` 실행 |
