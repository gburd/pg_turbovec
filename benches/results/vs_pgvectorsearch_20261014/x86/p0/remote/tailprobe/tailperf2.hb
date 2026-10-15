[heartbeat] starting: bash /home/fedora/tail_perf2.sh
[heartbeat] interval: 60s
[heartbeat] label:    bash
== scan_parallelism=8 n=1800 p50=6.9 p95=7.2
    40.47%  [.] crossbeam_epoch::default::with_handle::<crossbeam_epoch::default::pin::{closure#0}, crossbeam_epoch::guard::Guard>
    17.72%  [.] <crossbeam_epoch::internal::Global>::try_advance
     9.23%  [.] turbovec::search::search_single_query_vnni_blk2
     3.52%  [.] <crossbeam_deque::deque::Stealer<rayon_core::job::JobRef>>::steal
     2.28%  [.] <serde_cbor::de::Deserializer<serde_cbor::read::SliceRead>>::parse_value::<<f32 as serde_core::de::Deserialize>::deserialize::PrimitiveVisitor>
     2.18%  [.] __memset_avx512_unaligned_erms
     1.86%  [.] <std::sys::sync::mutex::futex::Mutex>::lock_contended
     1.63%  [.] __tls_get_addr
     1.56%  [.] turbovec::search::scan_with_luts::search_single_query_block_parallel::{closure#1}
     1.52%  [.] <serde_cbor::de::Deserializer<serde_cbor::read::SliceRead>>::recursion_checked::<<serde_cbor::de::Deserializer<serde_cbor::read::SliceRead>>::parse_array<<alloc::vec::Vec<_> as serde_core::de::Deserialize>::deserialize::VecVisitor<f32>>::{closure#0}, alloc::vec::Vec<f32>>
     1.50%  [.] <rayon_core::sleep::Sleep>::wake_specific_thread
     1.39%  [k] native_queued_spin_lock_slowpath
== scan_parallelism=1 n=1800 p50=6.9 p95=7.1
    39.35%  [.] crossbeam_epoch::default::with_handle::<crossbeam_epoch::default::pin::{closure#0}, crossbeam_epoch::guard::Guard>
    18.34%  [.] <crossbeam_epoch::internal::Global>::try_advance
     9.39%  [.] turbovec::search::search_single_query_vnni_blk2
     3.51%  [.] <crossbeam_deque::deque::Stealer<rayon_core::job::JobRef>>::steal
     2.30%  [.] <serde_cbor::de::Deserializer<serde_cbor::read::SliceRead>>::parse_value::<<f32 as serde_core::de::Deserialize>::deserialize::PrimitiveVisitor>
     2.21%  [.] __memset_avx512_unaligned_erms
     1.91%  [.] <std::sys::sync::mutex::futex::Mutex>::lock_contended
     1.69%  [.] __tls_get_addr
     1.53%  [.] <rayon_core::sleep::Sleep>::wake_specific_thread
     1.51%  [.] turbovec::search::scan_with_luts::search_single_query_block_parallel::{closure#1}
     1.43%  [.] <serde_cbor::de::Deserializer<serde_cbor::read::SliceRead>>::recursion_checked::<<serde_cbor::de::Deserializer<serde_cbor::read::SliceRead>>::parse_array<<alloc::vec::Vec<_> as serde_core::de::Deserialize>::deserialize::VecVisitor<f32>>::{closure#0}, alloc::vec::Vec<f32>>
     1.38%  [k] native_queued_spin_lock_slowpath
PERF2_DONE
[heartbeat] DONE rc=0 (clean exit)
