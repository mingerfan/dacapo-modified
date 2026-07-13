
if (HECATE_ENABLE_SEAL_RUNTIME)
  add_library (SEAL_HEVM
   SHARED
   SEAL_HEVM.cpp)
  target_link_libraries(SEAL_HEVM
   PUBLIC
   SEAL::seal
   )
endif()

# add_library (TOY_HEVM
#  SHARED
#  TOY_HEVM.cpp)


if (HECATE_ENABLE_HEAAN_RUNTIME)
  add_library (HEAAN_HEVM
    SHARED
    HEAAN_HEVM.cpp)
  target_link_libraries(HEAAN_HEVM
    PUBLIC
    HEaaN::HEaaN
    )
endif()
