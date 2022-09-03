pub struct ConstCheck<const CHECK: bool>;
pub trait True {}
impl True for ConstCheck<true> {}
