//! Tripwires: loss spike, dead token, SDC (spec 15.2 H12).

#[derive(Debug, PartialEq, Eq)]
pub enum Trip {
    LossSpike,
    DeadToken,
    Sdc,
}

pub fn watch(loss: f64, prev: f64, dead: u64, sdc_mismatch: bool) -> Option<Trip> {
    if sdc_mismatch {
        return Some(Trip::Sdc);
    }
    if dead > 0 {
        return Some(Trip::DeadToken);
    }
    if prev > 0.0 && loss > prev * 2.0 {
        return Some(Trip::LossSpike);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trips() {
        assert_eq!(watch(1.0, 1.0, 0, true), Some(Trip::Sdc));
        assert_eq!(watch(1.0, 1.0, 2, false), Some(Trip::DeadToken));
        assert_eq!(watch(5.0, 2.0, 0, false), Some(Trip::LossSpike));
        assert_eq!(watch(1.1, 1.0, 0, false), None);
    }
}
