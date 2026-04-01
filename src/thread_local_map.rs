//! This module provides a thread-local map for storing and retrieving type-indexed data.
//!
//! It is useful for caching objects that are expensive to create, such as FFT planners,
//! in a thread-safe manner without requiring global locks.

use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::HashMap;

/// Retrieves an item of type `T` from the provided map, or initializes it if it doesn't exist.
///
/// This function removes the item from the map before passing it to the `with` closure, and
/// puts it back afterward. This allows re-entrant access to the map for *different* types
/// without causing deadlocks or panics.
pub fn get_or_init_with<T: 'static, R>(
    map: &RefCell<HashMap<TypeId, Box<dyn Any>>>,
    init: impl FnOnce() -> T,
    with: impl FnOnce(&T) -> R,
) -> R {
    let type_id = TypeId::of::<T>();

    // We remove the item from the map to avoid holding a Ref mut over the `with` call.
    // This allows `with` to re-entrantly access the map for OTHER types.
    let item: Box<dyn Any> = map.borrow_mut().remove(&type_id).unwrap_or_else(|| Box::new(init()));

    // Downcast to the actual type
    let item_ref = item.downcast_ref::<T>().unwrap();
    let result = with(item_ref);

    // Put it back
    map.borrow_mut().insert(type_id, item);
    result
}

/// A macro for getting or initializing an item in a thread-local map.
///
/// This macro initializes a thread-local map on first use and provides a safe interface
/// to access it re-entrantly for different types.
#[macro_export]
macro_rules! get_or_init_thread_local {
    ($init:expr, $with:expr) => {{
        thread_local! {
            static MAP: std::cell::RefCell<std::collections::HashMap<std::any::TypeId, Box<dyn std::any::Any>>> = std::cell::RefCell::new(std::collections::HashMap::new());
        }
        MAP.with(|map| $crate::thread_local_map::get_or_init_with(map, $init, $with))
    }};
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    fn with_cell<T: 'static, R>(init: i32, f: impl FnOnce(&Cell<i32>) -> R) -> R {
        crate::get_or_init_thread_local!(|| Cell::new(init), f)
    }

    #[test]
    fn test_basic_initialization() {
        with_cell::<i32, _>(0, |cell| {
            cell.set(cell.get() + 1);
            assert_eq!(cell.get(), 1);
        });

        with_cell::<i32, _>(0, |cell| {
            cell.set(cell.get() + 1);
            assert_eq!(cell.get(), 2);
        });
    }

    #[test]
    fn test_different_types() {
        with_cell::<i32, _>(0, |cell_i32| {
            cell_i32.set(1);
            with_cell::<f32, _>(0, |cell_f32| {
                cell_f32.set(2);
                assert_eq!(cell_f32.get(), 2);
            });
            assert_eq!(cell_i32.get(), 1);
        });
    }

    #[test]
    fn test_reentrancy_different_types_does_not_panic() {
        with_cell::<i32, _>(10, |cell1| {
            with_cell::<f32, _>(20, |cell2| {
                assert_eq!(cell1.get(), 10);
                assert_eq!(cell2.get(), 20);
            });
        });
    }

    #[test]
    fn test_same_type_reentrancy_initializes_twice_and_overwrites() {
        with_cell::<u32, _>(1, |outer_cell| {
            outer_cell.set(2);
            with_cell::<u32, _>(10, |inner_cell| {
                // inner_cell sees the new initialization
                assert_eq!(inner_cell.get(), 10);
                inner_cell.set(20);
            });
            // Outer cell retains its state
            assert_eq!(outer_cell.get(), 2);
        });

        // After outer completes, it overwrites the map with outer_cell's state,
        // so the inner's value (20) is lost, reverting to 2.
        with_cell::<u32, _>(100, |cell| {
            assert_eq!(cell.get(), 2);
        });
    }

    #[test]
    fn test_thread_isolation() {
        use std::thread;

        with_cell::<i32, _>(10, |cell| {
            cell.set(20);
            assert_eq!(cell.get(), 20);

            // Spawn a thread that should see its own separate value
            let handle = thread::spawn(|| {
                with_cell::<i32, _>(100, |inner_cell| {
                    assert_eq!(inner_cell.get(), 100);
                    inner_cell.set(200);
                    assert_eq!(inner_cell.get(), 200);
                });
            });

            handle.join().unwrap();

            // Main thread's value should still be 20
            assert_eq!(cell.get(), 20);
        });
    }

    #[test]
    fn test_panic_safety_reinitialization() {
        use std::panic;

        // Verify that if a `with` closure panics, the item is removed but re-initialized next time.
        let result = panic::catch_unwind(|| {
            with_cell::<i32, _>(5, |cell| {
                cell.set(50);
                panic!("Intentional panic inside with closure");
            });
        });

        assert!(result.is_err());

        // Next call should re-initialize with the default value (5) instead of failing or being poisoned.
        with_cell::<i32, _>(5, |cell| {
            assert_eq!(cell.get(), 5);
        });
    }
}
