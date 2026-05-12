target_sources(nn2logic PRIVATE
    Encoder.hpp
    bindings.cpp
    IndicatorWorklist.hpp
    MultiProgress.hpp
    Settings.hpp
    Path.hpp
    Path.cpp
    StorageAdaptor.hpp)

add_subdirectory(analytics)
add_subdirectory(quant)  # todo
add_subdirectory(codegen)
add_subdirectory(codegen2)
add_subdirectory(sim)