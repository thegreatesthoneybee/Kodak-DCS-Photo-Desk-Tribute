# Kodak DCS Photo Desk Tribute

**PhotoDesk (Tribute)** brings Kodak DCS Photo Desk 4.3 back to modern Windows and Apple Silicon Macs.
It is a native, cross-platform reimplementation for opening and developing Kodak DCS raw files (`.DCR`)
with the same interface, the same workflow and Kodak's original colour profiles.

[한국어 안내는 아래에 있습니다.](#한국어)

## Download

Get the latest build from [Releases](../../releases/latest).

| Platform | File |
|---|---|
| Windows 10/11 (x64) | `PhotoDesk-Tribute-<version>-windows-x64.zip` |
| macOS 11+ (Apple Silicon) | `PhotoDesk-Tribute-<version>-macos-arm64.zip` |

`SHA256SUMS.txt` in each release lists the checksums of the files.

### Windows

1. Unzip the file anywhere (for example `C:\Program Files\PhotoDesk` or your Desktop).
2. Run `PhotoDesk\PhotoDesk.exe`. Keep the `_internal` folder next to it.
3. Windows SmartScreen may warn about an unknown publisher on first start. Choose **More info → Run anyway**.

### macOS

1. Unzip the file and move `PhotoDesk.app` to **Applications**.
2. The app is not notarized by Apple. On first start, **right-click (Control-click) the app → Open → Open**.
   If macOS says the app is damaged, run this once in Terminal:
   ```
   xattr -dr com.apple.quarantine /Applications/PhotoDesk.app
   ```

## Features

- The original interface: menus, toolbar, contact sheets, image windows, image controls and info panels,
  dialogs, cursors and icons, in all 10 original languages (including Korean).
- Kodak colour processing with the original ICC profiles: camera/lighting source corrections, ERIMM,
  Product/Portrait Looks and RGB working spaces.
- Exposure, tone, white balance (preset/click/auto), colour balance, white point, noise reduction,
  sharpening, crop and rotation; settings copy/paste and undo/redo.
- JPEG/TIFF export with IPTC and EXIF (including GPS), printing and contact sheets, white balance (.wb) files.
- Batch conversion from the command line: `PhotoDesk convert <files|folders> -o <output folder>`.

The original help files are not included, so **Help Topics** reports that no help file is available.

## Disclaimer

This is an unofficial, non-commercial tribute made by a Kodak DCS user. It is not affiliated with,
endorsed by or supported by Eastman Kodak Company. Kodak, DCS and Photo Desk are trademarks of their
respective owners.

Kodak DCS Photo Desk was discontinued and is no longer offered on Kodak's website. To keep the original
colour rendition, the builds contain Kodak's colour profiles, the official translation table and the
interface resources (dialog layouts, menus, bitmaps and strings) of the original program. The original
program code and help files are not included.

The software is provided "as is", without warranty of any kind. Keep copies of your original raw files.

Created by GO BANGWON (Korea).

---

## 한국어

**PhotoDesk (Tribute)** 는 Kodak DCS Photo Desk 4.3을 최신 Windows와 Apple Silicon Mac에서 다시 쓸 수 있도록
새로 구현한 프로그램입니다. Kodak DCS raw 파일(`.DCR`)을 원본과 같은 화면, 같은 작업 방식, Kodak 원본 색 프로파일로
열고 현상합니다.

### 다운로드

[Releases](../../releases/latest)에서 최신 버전을 받습니다.

| 플랫폼 | 파일 |
|---|---|
| Windows 10/11 (x64) | `PhotoDesk-Tribute-<버전>-windows-x64.zip` |
| macOS 11 이상 (Apple Silicon) | `PhotoDesk-Tribute-<버전>-macos-arm64.zip` |

릴리즈마다 있는 `SHA256SUMS.txt`에 파일 체크섬이 있습니다.

### Windows

1. 원하는 곳에 압축을 풉니다.
2. `PhotoDesk\PhotoDesk.exe`를 실행합니다. 같은 폴더의 `_internal` 폴더는 지우거나 옮기지 마세요.
3. 처음 실행할 때 SmartScreen 경고가 나오면 **추가 정보 → 실행**을 누릅니다.

### macOS

1. 압축을 풀고 `PhotoDesk.app`을 **응용 프로그램** 폴더로 옮깁니다.
2. Apple 공증을 받지 않은 앱이라, 처음에는 **앱을 우클릭(Control-클릭) → 열기 → 열기**로 실행합니다.
   "손상되었습니다"라고 나오면 터미널에서 한 번 실행합니다.
   ```
   xattr -dr com.apple.quarantine /Applications/PhotoDesk.app
   ```

### 기능

- 원본과 같은 화면: 메뉴, 툴바, 컨택트 시트, 이미지 창, 이미지 조절·정보 패널, 대화상자, 커서, 아이콘. 원본의 10개 언어(한국어 포함).
- 원본 Kodak ICC 프로파일로 색 처리: 카메라·광원별 Source Correction, ERIMM, Product/Portrait Look, RGB 작업 색 공간.
- 노출, 톤, 화이트 밸런스(사전 설정/클릭/자동), 컬러 밸런스, 화이트포인트, 노이즈 제거, 샤프닝, 트리밍, 회전. 설정 복사/붙여넣기, 실행 취소/다시 실행.
- JPEG/TIFF 내보내기(IPTC, GPS 포함 EXIF), 인쇄와 컨택트 시트, 화이트 밸런스(.wb) 파일.
- 명령줄 일괄 변환: `PhotoDesk convert <파일|폴더> -o <출력 폴더>`.

원본 도움말은 포함하지 않아 **도움말 항목**을 누르면 도움말 파일이 없다고 나옵니다.

### 면책

Kodak DCS 사용자가 만든 비공식·비상업 트리뷰트입니다. Eastman Kodak Company와 관계가 없으며, Kodak의 승인이나 지원을
받지 않았습니다. Kodak, DCS, Photo Desk는 각 소유자의 상표입니다.

Kodak DCS Photo Desk는 단종되어 지금은 Kodak 홈페이지에서도 받을 수 없습니다. 원본과 같은 색을 지키기 위해 Kodak 색 프로파일,
공식 번역표, 원본 프로그램의 화면 리소스(대화상자 배치, 메뉴, 비트맵, 문자열)를 포함했습니다. 원본 프로그램 코드와 도움말은
포함하지 않았습니다.

이 소프트웨어는 어떠한 보증도 없이 "있는 그대로" 제공됩니다. 원본 raw 파일은 따로 보관하세요.
