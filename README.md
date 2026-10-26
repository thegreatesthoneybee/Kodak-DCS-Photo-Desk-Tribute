# PhotoDesk

Kodak DCS Photo Desk 4.3의 RAW 현상 기능을 Windows 11(x64)과 macOS(Apple Silicon)에서
네이티브로 다시 구현한 프로그램입니다.

- RAW 디코딩: LibRaw (rawpy)
- 색 처리: Photo Desk 4.3에 들어 있던 Kodak ICC 프로파일을 그대로 사용
  `카메라 RGB → Source Correction(카메라/광원) → ERIMM → Look → sRGB / Adobe RGB / ProPhoto RGB`
- 컨택트 시트, 리뷰 화면, 광원·Look·화이트밸런스(회색 클릭)·노출·샤프닝·노이즈 감소·회전
- 설정 복사/붙여넣기 (Photo Desk의 Paste Special), 사이드카 파일 `<파일>.photodesk.json`에 저장
- JPEG(EXIF·ICC 포함), TIFF 8/16비트 내보내기, 긴 변 리사이즈, 일괄 변환 CLI

## Kodak 프로파일 설치 (최초 1회)

Kodak 프로파일은 저작권 문제로 함께 배포하지 않습니다. 원본 설치 파일
`Photo_Desk_V_4_3_0.exe`를 지정하면, 실행하지 않고 내부 CAB/InstallShield 아카이브에서
프로파일 57개만 추출합니다. macOS에서도 같은 .exe 파일을 사용합니다.

- GUI: `File > Install Kodak Profiles...`
- CLI: `photodesk install-profiles Photo_Desk_V_4_3_0.exe`

설치 위치: Windows `%APPDATA%\PhotoDesk\profiles`, macOS `~/Library/Application Support/PhotoDesk/profiles`
(환경변수 `PHOTODESK_PROFILES`로 변경 가능)

## 사용

```
photodesk                         # GUI
photodesk convert <파일|폴더...> -o <출력폴더> [-f JPEG|"TIFF 8-bit"|"TIFF 16-bit"]
          [-s sRGB|"Adobe RGB (1998)"|"ProPhoto RGB"] [--look DCSPortraitLook.icm]
          [--illuminant Daylight|Flash|Tungsten|Fluorescent] [--long-edge 3000]
photodesk looks                   # 설치된 Look 목록
```

## 카메라 지원

Kodak Source Correction 프로파일이 있는 기종: DCS Pro Back / Pro Back 645, DCS Pro 14n·14nx,
Pro SLR/n, Pro SLR/c, DCS 720x, DCS 760. 그 밖에 LibRaw가 읽는 기종(DCS 3xx/5xx/6xx 등)은
LibRaw 색 매트릭스로 ERIMM을 만든 뒤 Kodak Look을 적용합니다.

Kodak 파일에 기록된 촬영 시 화이트밸런스는 LibRaw가 잘못 읽습니다(중성 회색이 청록색으로 나옴).
그래서 Kodak 파일은 기본 화이트밸런스를 `Daylight`로 시작합니다. `As Shot`, `Auto`,
회색 클릭(`Pick Neutral`)으로 바꿀 수 있습니다.

## 개발 / 빌드

```
pip install -e .[dev]
python -m pytest            # vendor/original/Photo_Desk_V_4_3_0.exe, raw/*.DCR가 있으면 전체 테스트 실행
pyinstaller packaging/photodesk.spec
```

Windows x64 빌드는 Windows에서, macOS arm64 빌드(`PhotoDesk.app`)는 Apple Silicon Mac에서 만듭니다.
`.github/workflows/build.yml`이 두 플랫폼 빌드를 모두 수행합니다(macOS는 `macos-14` M1 러너).
