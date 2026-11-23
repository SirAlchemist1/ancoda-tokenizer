# Unicode 16.0.0 UCD file hashes (setup step 8)

Downloaded from `https://www.unicode.org/Public/16.0.0/ucd/` into `third_party/unicode/16.0.0/` (gitignored).

| File | sha256 |
|---|---|
| DerivedCoreProperties.txt | 39d35161f2954497f69e08bdb9e701493f476a3d30222de20028feda36c1dabd |
| Scripts.txt | 9e88f0a677df47311106340be8ede2ecdacd9c1c931831218d2be6d5508e0039 |
| UnicodeData.txt | ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f |
| auxiliary/GraphemeBreakTest.txt | ee2b9354d270ac061b29f09662cafea06341d77e704b8cc6bd72aaeeda363cb5 |
| auxiliary/GraphemeBreakProperty.txt | c29360bd6f7132811d701d29069541e827eb44bfc4c8fbde8c370d6982689dc1 |
| emoji/emoji-data.txt | f1365a5173eee18e1f98b240cdc492e84a25f1ce7e0c9d1094eb29c41a22696a |

pretokenizer added `GraphemeBreakProperty.txt` and `emoji/emoji-data.txt` for the independent Python segmenter (UAX #29). `emoji-data.txt` is the Unicode 16.0.0 file dated 2024-05-01; only the `Extended_Pictographic` property is read.

`unicode-segmentation` 1.12.0 `UNICODE_VERSION` assert: **(16, 0, 0)** — `cargo test -p ancoda-seg unicode_version_is_16_0_0` passed.
