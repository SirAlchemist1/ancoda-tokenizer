//! Sorted pre-token count table.
//!
//! ```text
//! magic: b"ANCDACT1"
//! u64le entry_count
//! repeated, sorted by bytes ascending:
//!   u32le byte_length
//!   bytes
//!   u64le count
//! ```

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"ANCDACT1";

pub struct CountTable {
    pub entries: Vec<(Vec<u8>, u64)>,
    pub occurrences: u64,
    pub key_bytes: u64,
}

pub fn read_counts(path: &Path) -> Result<CountTable, String> {
    let mut file = File::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)
        .map_err(|err| format!("read magic: {err}"))?;
    if &magic != MAGIC {
        return Err(format!("bad count magic in {}", path.display()));
    }
    let mut buf8 = [0u8; 8];
    file.read_exact(&mut buf8)
        .map_err(|err| format!("read count header: {err}"))?;
    let n = u64::from_le_bytes(buf8);
    let mut entries = Vec::with_capacity(usize::try_from(n).map_err(|err| err.to_string())?);
    let mut occurrences = 0u64;
    let mut key_bytes = 0u64;
    let mut previous: Option<Vec<u8>> = None;
    for index in 0..n {
        let mut buf4 = [0u8; 4];
        file.read_exact(&mut buf4)
            .map_err(|err| format!("read len {index}: {err}"))?;
        let len = u32::from_le_bytes(buf4) as usize;
        let mut bytes = vec![0u8; len];
        if len > 0 {
            file.read_exact(&mut bytes)
                .map_err(|err| format!("read bytes {index}: {err}"))?;
        }
        file.read_exact(&mut buf8)
            .map_err(|err| format!("read freq {index}: {err}"))?;
        let count = u64::from_le_bytes(buf8);
        if count == 0 {
            return Err(format!("zero count at entry {index}"));
        }
        if let Some(prev) = &previous {
            if bytes.as_slice() <= prev.as_slice() {
                return Err(format!("counts are not strictly sorted at entry {index}"));
            }
        }
        occurrences = occurrences
            .checked_add(count)
            .ok_or("occurrence count overflow")?;
        key_bytes = key_bytes
            .checked_add(len as u64)
            .ok_or("key byte overflow")?;
        previous = Some(bytes.clone());
        entries.push((bytes, count));
    }
    Ok(CountTable {
        entries,
        occurrences,
        key_bytes,
    })
}

pub fn write_counts(path: &Path, entries: &[(Vec<u8>, u64)]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    let mut file = File::create(path).map_err(|err| err.to_string())?;
    file.write_all(MAGIC).map_err(|err| err.to_string())?;
    file.write_all(&(entries.len() as u64).to_le_bytes())
        .map_err(|err| err.to_string())?;
    let mut previous: Option<&[u8]> = None;
    for (bytes, count) in entries {
        if let Some(prev) = previous {
            if bytes.as_slice() <= prev {
                return Err("refusing to write unsorted counts".into());
            }
        }
        previous = Some(bytes.as_slice());
        file.write_all(&(bytes.len() as u32).to_le_bytes())
            .map_err(|err| err.to_string())?;
        file.write_all(bytes).map_err(|err| err.to_string())?;
        file.write_all(&count.to_le_bytes())
            .map_err(|err| err.to_string())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn round_trip_sorted_counts() {
        let path = env::temp_dir().join("ancoda-counts-roundtrip.bin");
        let entries = vec![
            (b"a".to_vec(), 2u64),
            (b"ab".to_vec(), 5),
            ("क".as_bytes().to_vec(), 7),
        ];
        write_counts(&path, &entries).unwrap();
        let table = read_counts(&path).unwrap();
        assert_eq!(table.entries, entries);
        assert_eq!(table.occurrences, 14);
        let _ = std::fs::remove_file(&path);
    }
}
