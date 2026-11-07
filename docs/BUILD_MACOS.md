# macOS (Apple Silicon / Intel) 빌드 가이드

Apple Silicon에서는 arm64 Python으로, Intel Mac에서는 x86_64 Python으로 `PhotoDesk.app`을 빌드합니다. 빌드 프로세스의 아키텍처를 사용하며 두 배포 파일은 별개입니다.
Apple Silicon Mac에서도 Rosetta가 설치되어 있으면 x86_64 Python과 x86_64 의존성으로 Intel용 앱을 빌드하고 실행 검증할 수 있습니다. `.github/workflows/build.yml`의 `macos-14` 작업은 Apple Silicon용입니다.

## 1. 준비물

| 항목 | 내용 |
|---|---|
| macOS | 앱에 포함되는 Python·Qt 등 의존성의 최소 OS 요구사항을 충족해야 함 |
| Python | 3.10 이상, 배포 대상에 맞는 **arm64 또는 x86_64 빌드** |
| Xcode 명령행 도구 | `xcode-select --install` (codesign, ditto 사용) |
| 원본 설치 파일 | `Photo_Desk_V_4_3_0.exe` (프로파일·UI·도움말 설치용, 앱에 포함하지 않음) |
| 테스트용 RAW | 선택 사항. `raw/*.DCR` |

Python 아키텍처를 먼저 확인합니다. Apple Silicon용은 `arm64`, Intel용은 `x86_64`가 나와야 합니다. Rosetta에서 Intel용을 빌드할 때도 전체 의존성을 별도 x86_64 가상환경에 설치해야 합니다.

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

결과물은 `dist/PhotoDesk.app`입니다. spec 파일은 실행 중인 Python의 아키텍처로 빌드하고 `BUNDLE`로 .app을 만듭니다. 번들 ID는 `org.photodesk.PhotoDesk`입니다. 다른 아키텍처의 기존 결과를 보존하려면 `--distpath dist/macos-x86_64 --workpath build/macos-x86_64`처럼 별도 경로를 사용하세요. Info.plist의 기본 OS 표기만으로 호환성을 판단하지 말고, 포함된 Mach-O 라이브러리들의 최소 OS 요구사항도 확인해야 합니다.

빌드 확인:

```bash
file dist/PhotoDesk.app/Contents/MacOS/PhotoDesk        # 대상에 따라 "arm64" 또는 "x86_64"
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
- **검증 상태:** Windows 실행 및 M2 Pro의 arm64 빌드·서명·CLI를 확인했습니다. v0.5.5 Intel용은 M2 Pro의 Rosetta에서 x86_64 Python/LibRaw를 사용해 테스트·RAW 변환·GUI 기동·압축 해제 후 서명과 CLI를 검증했습니다. 물리적인 Intel Mac에서의 실행은 확인하지 않았습니다.
- **Intel v0.5.5 배포 요구사항:** 포함된 Mach-O 의존성의 최소 OS 요구사항을 검사해 macOS 15.0 이상으로 표시했습니다. Python 3.13.18 x86_64, rawpy 0.27.0/LibRaw 0.22.1 소스 빌드, PySide6 6.10.2를 사용합니다. `PhotoDesk-Tribute-0.5.5-macos-x86_64.zip`을 선택하세요.

## 문제 해결

| 증상 | 원인과 해결 |
|---|---|
| `file`의 아키텍처가 배포 대상과 다름 | 대상과 같은 아키텍처의 Python으로 별도 가상환경을 구성하고 재빌드 |
| Intel용 rawpy 0.27 휠을 찾지 못함 | 공식 rawpy v0.27.0 소스와 해당 Git 하위 모듈의 LibRaw를 x86_64로 빌드해야 함. 기능 요구사항 때문에 구버전으로 낮추지 않음 |
| Rosetta 빌드에서 libxcrun의 x86_64 아키텍처가 없다는 오류 | 최신 Command Line Tools는 arm64 전용일 수 있음. 실제 CLT clang/make 경로를 사용하고 CMake의 `CMAKE_OSX_ARCHITECTURES=x86_64`와 명시적 SDK 경로로 크로스 컴파일 |
| 앱이 바로 꺼짐 | 터미널에서 `dist/PhotoDesk.app/Contents/MacOS/PhotoDesk`를 직접 실행해 오류 확인 |
| "Kodak profiles not installed" | 4절의 `install-profiles` 실행 후 다시 빌드 |
| 앱이 "손상되었습니다"로 열리지 않음 | 격리 속성 문제. 7절의 `xattr -dr com.apple.quarantine` 실행 |
