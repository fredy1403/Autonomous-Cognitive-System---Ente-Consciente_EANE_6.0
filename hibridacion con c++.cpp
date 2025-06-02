// se actualizara el sistema eane phoenix con este codigo laimplementacion sera unicamente para una hibridacion donde solo los modulos que mas se vean beneficiados por c++ se cambiaran a c++ de manera que toda la arquitectura seguira funcionando en python y solo se usara a c++ en donde se vea necesario su uso

// CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(EanePhoenixCpp VERSION 16.0) // Updated version

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # For shared libraries

# --- Compiler Flags (Optional: Add for more warnings/optimizations) ---
# if(MSVC)
#   add_compile_options(/W4 /WX) # Example for MSVC
# else()
#   add_compile_options(-Wall -Wextra -pedantic -Werror) # Example for GCC/Clang
# endif()

# --- Eigen ---
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR}) # Use SYSTEM for external libraries

# --- Pybind11 ---
# Using FetchContent is a robust way to manage pybind11 dependency
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0 # Or a more recent stable tag
)
FetchContent_MakeAvailable(pybind11) # This makes pybind11 targets available

# --- Source Directories ---
# Define base source directory for cleaner paths
set(EANE_CPP_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/eane_cpp_modules)

# --- Include Directories ---
include_directories(
    ${EANE_CPP_SOURCE_DIR} # For core_interface.h etc. at top level of eane_cpp_modules
    # Subdirectories will be added as needed by modules, or can be listed here
    # e.g., ${EANE_CPP_SOURCE_DIR}/mathematical_toolkit
)

# --- Source File Definitions ---
# Group sources by logical components for better organization

# Core Utilities
set(CORE_UTILS_SRC
    # core_interface.cpp (if it had one, currently header-only concept)
)

# Mathematical Toolkit & Simulators
set(MATH_TOOLKIT_SRC
    ${EANE_CPP_SOURCE_DIR}/mathematical_toolkit/mathematical_toolkit.cpp
)
set(PHYSICS_SIMULATORS_SRC
    ${EANE_CPP_SOURCE_DIR}/physics_simulators/physics_simulators.cpp # Combined header
    ${EANE_CPP_SOURCE_DIR}/physics_simulators/quantum_mechanics_simulator.cpp
    ${EANE_CPP_SOURCE_DIR}/physics_simulators/cosmology_simulator.cpp
    ${EANE_CPP_SOURCE_DIR}/physics_simulators/stochastic_simulator.cpp
)

# Foundational Cognitive Modules
set(SUBCORE_SRC
    ${EANE_CPP_SOURCE_DIR}/subconscious_mind/subconscious_mind.cpp
)
set(LEARNING_MODULE_SRC
    ${EANE_CPP_SOURCE_DIR}/learning_module/learning_module.cpp
    ${EANE_CPP_SOURCE_DIR}/learning_module/lstm_stub.cpp
    ${EANE_CPP_SOURCE_DIR}/learning_module/q_learning_agent_stub.cpp
    ${EANE_CPP_SOURCE_DIR}/learning_module/knowledge_base_stub.cpp
)
set(SEM_SRC
    ${EANE_CPP_SOURCE_DIR}/self_evolution_module/self_evolution_module.cpp
    # sem_types.cpp (if created)
)
set(FREEWILL_MODULE_SRC
    ${EANE_CPP_SOURCE_DIR}/freewill_module/freewill_module.cpp
    # freewill_types.cpp (if created)
)
set(FREEWILL_ENGINE_SRC
    ${EANE_CPP_SOURCE_DIR}/freewill_engine/freewill_engine.cpp
    ${EANE_CPP_SOURCE_DIR}/freewill_engine/environment_fwe.cpp
)

# Interface & Support Modules
set(FIREWALL_SRC
    ${EANE_CPP_SOURCE_DIR}/adaptive_firewall_module/adaptive_firewall_module.cpp
    ${EANE_CPP_SOURCE_DIR}/adaptive_firewall_module/firewall_types.cpp
)
set(TS_PREDICTOR_SRC
    ${EANE_CPP_SOURCE_DIR}/timeseries_predictor_module/timeseries_predictor_module.cpp
    # ts_types.cpp (if created)
)

# Advanced Processing & Lyuk
set(MUGEN_SRC
    ${EANE_CPP_SOURCE_DIR}/controlled_mutation_generator/controlled_mutation_generator.cpp
    # mugen_types.cpp (if created)
)
set(LYUK_PARSER_SRC
    ${EANE_CPP_SOURCE_DIR}/lyuk_parser/lyuk_parser.cpp
    # lyuk_ast_types.cpp (if created)
)

# Homeostasis & Regulation Stubs
set(CONSCIOUSNESS_MODULE_STUB_SRC ${EANE_CPP_SOURCE_DIR}/consciousness_module_cpp/consciousness_module.cpp)
set(EMOTION_REGULATION_STUB_SRC ${EANE_CPP_SOURCE_DIR}/emotion_regulation_module_cpp/emotion_regulation_module.cpp)
set(NEEDS_MANAGER_STUB_SRC ${EANE_CPP_SOURCE_DIR}/needs_manager_cpp/needs_manager.cpp)
set(CRAVING_MODULE_STUB_SRC ${EANE_CPP_SOURCE_DIR}/craving_module_cpp/craving_module.cpp)
set(SELF_COMPASSION_STUB_SRC ${EANE_CPP_SOURCE_DIR}/self_compassion_module_cpp/self_compassion_module.cpp)
set(STRESS_RESPONSE_STUB_SRC ${EANE_CPP_SOURCE_DIR}/stress_response_module_cpp/stress_response_module.cpp)
set(PAIN_MATRIX_STUB_SRC ${EANE_CPP_SOURCE_DIR}/pain_matrix_directive_cpp/pain_matrix_directive.cpp)
set(DEFENSE_MECH_STUB_SRC ${EANE_CPP_SOURCE_DIR}/defense_mechanisms_cpp/defense_mechanisms.cpp)

# Communication & Social Stubs
set(LLYUK_COMM_STUB_SRC ${EANE_CPP_SOURCE_DIR}/llyuk_communication_module_cpp/llyuk_communication_module.cpp)
set(SOCIAL_DYNAMICS_STUB_SRC ${EANE_CPP_SOURCE_DIR}/social_dynamics_module_cpp/social_dynamics_module.cpp)
set(ONTOLOGY_FLOW_STUB_SRC ${EANE_CPP_SOURCE_DIR}/ontology_flow_manager_cpp/ontology_flow_manager.cpp)


# Pybind11 Wrapper Source
set(PYBIND_WRAPPER_SRC ${EANE_CPP_SOURCE_DIR}/pybind_wrapper.cpp)

# --- Python Module Target ---
pybind11_add_module(eane_cpp_core SHARED
    ${PYBIND_WRAPPER_SRC}
    ${CORE_UTILS_SRC}
    ${MATH_TOOLKIT_SRC}
    ${PHYSICS_SIMULATORS_SRC}
    ${SUBCORE_SRC}
    ${LEARNING_MODULE_SRC}
    ${SEM_SRC}
    ${FREEWILL_MODULE_SRC}
    ${FREEWILL_ENGINE_SRC}
    ${FIREWALL_SRC}
    ${TS_PREDICTOR_SRC}
    ${MUGEN_SRC}
    ${LYUK_PARSER_SRC}
    ${CONSCIOUSNESS_MODULE_STUB_SRC}
    ${EMOTION_REGULATION_STUB_SRC}
    ${NEEDS_MANAGER_STUB_SRC}
    ${CRAVING_MODULE_STUB_SRC}
    ${SELF_COMPASSION_STUB_SRC}
    ${STRESS_RESPONSE_STUB_SRC}
    ${PAIN_MATRIX_STUB_SRC}
    ${DEFENSE_MECH_STUB_SRC}
    ${LLYUK_COMM_STUB_SRC}
    ${SOCIAL_DYNAMICS_STUB_SRC}
    ${ONTOLOGY_FLOW_STUB_SRC}
    # Add other SRC variables here as modules are implemented
)

# --- Linking ---
# Eigen is mostly header-only, but if specific parts need linking:
# target_link_libraries(eane_cpp_core PRIVATE Eigen3::Eigen)

# --- RPATH settings for non-Windows systems ---
if(UNIX AND NOT APPLE)
    set_target_properties(eane_cpp_core PROPERTIES
        INSTALL_RPATH "$ORIGIN" # RPATH relative to the executable/library
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# --- Installation (Optional) ---
# install(TARGETS eane_cpp_core LIBRARY DESTINATION your_python_site_packages_path) # Adjust path

# --- Testing (Optional, using CTest) ---
# enable_testing()
# add_test(NAME MyCppTest COMMAND MyTestExecutable)
# target_link_libraries(MyTestExecutable PRIVATE eane_cpp_core) # If tests link against the module
// eane_cpp_modules/core_interface.h
#pragma once

#include <string>
#include <vector>
#include <map>
#include <any>        // For generic content in events, though use with caution
#include <variant>    // C++17, safer alternative to std::any for known types
#include <optional>   // C++17
#include <functional> // For callbacks or strategy patterns if needed
#include <chrono>     // For timestamps

// Forward declare Eigen types if they are part of the interface but Eigen headers are heavy
// However, for GlobalSelfStateCpp, it's better to include Eigen directly.
#include <Eigen/Dense>

// --- Global Self State (C++ Mirror) ---
// This struct mirrors the Python GlobalSelfState.
// Synchronization between Python and C++ states is a major challenge
// and typically handled by the CoreInterface implementation.
struct GlobalSelfStateCpp {
    // Emotional & Motivational
    double valencia = 0.0;
    double arousal = 0.5;
    double motivacion = 0.5;
    double dolor = 0.0;

    // Psychological Needs (Autonomy, Relatedness, Competence)
    Eigen::Vector3d needs_vector; // Fixed size 3

    // Beliefs (Simplified probability distribution)
    Eigen::VectorXd beliefs_distribution; // Size can be dynamic or fixed

    // Cognitive & Consciousness Metrics
    double phi_consciousness = 0.0;
    double phi_funcional_score = 0.0;
    double coherence_score = 0.75;
    double synchrony_metric = 0.7; // Renamed from 'synchrony' to avoid conflict if it's a keyword
    double system_entropy = 0.12;

    // Self-Perception & Narrative
    double self_esteem = 0.7;
    std::string qualia_state_label = "neutral_adaptativo_cpp";
    // std::any narrative_self_ref_for_cm_cpp; // Too complex for direct C++/Python bridging easily

    // Values & Ethics
    std::map<std::string, double> values;

    // Goals & Decisions (Simplified for C++)
    // For complex goal structures, dedicated structs would be needed.
    std::map<std::string, std::map<std::string, std::any>> goals_map_cpp; // goal_id -> {description: string, priority: double, ...}
    std::map<std::string, std::any> current_top_goal_cpp;
    std::map<std::string, std::any> last_decision_cpp;
    std::map<std::string, std::any> current_focus_info_cpp;

    // System & Time Parameters
    std::string system_id_tag = "EnteConsciente_Riku_Phoenix_V16.0_CppPart";
    double system_timestamp = 0.0; // Typically synchronized from Python Core
    double time_delta_continuous_step = 0.1;

    // Stability & Risk
    double system_threat_level_value = 0.05;
    double resilience_stability_metric = 0.9;

    // Rhythms & Activity
    double circadian_activity_level_value = 0.6;
    // std::string active_module_combination_id_cpp; // Complex to manage across boundary
    // std::map<std::string, bool> module_sleep_states_cpp;

    // MuGen/SEM specific (conceptual representations)
    // Eigen::VectorXd system_context_vector_for_mugen_sim_cpp;
    // std::map<std::string, std::any> active_fitness_landscape_config_for_sem_cpp;

    GlobalSelfStateCpp() : needs_vector(0.7, 0.7, 0.7), beliefs_distribution(Eigen::Vector3d(1.0/3.0, 1.0/3.0, 1.0/3.0)) {
        values = {
            {"no_dañar_intencionalmente_v2_cpp", 0.9},
            {"promover_bienestar_consciente_v2_cpp", 0.8},
            {"mantener_integridad_eane_v2_cpp", 0.95},
            {"evolucion_consciente_adaptativa_v2_cpp", 0.9}
            // Add more default values as in Python
        };
    }
};

// --- Event System (C++ Mirror) ---
// Using std::variant for safer and more explicit event content types.
// Add more types to the variant as needed by C++ modules.
using EventContentValueCpp = std::variant<
    std::monostate, // Represents no value or an empty content field
    bool,
    int,
    long long, // For larger integers if needed
    double,
    std::string,
    Eigen::VectorXd, // For numerical vectors
    std::vector<double>, // Alternative for simple numerical lists
    std::map<std::string, double>, // For key-value numerical data
    std::map<std::string, std::string> // For key-value string data
    // Consider adding std::vector<std::string>
    // For more complex data, pass IDs or use JSON strings, or define specific structs and add them to the variant.
>;

struct EventDataCpp {
    std::string type;
    std::map<std::string, EventContentValueCpp> content; // Key-value store for event payload
    std::string priority_label = "medium"; // "low", "medium", "high", "critical"
    std::optional<std::string> source_module;
    std::optional<std::string> target_module; // For direct inter-module C++ events (if used)
    double timestamp_creation = 0.0; // Timestamp of event creation

    EventDataCpp(std::string t = "") : type(std::move(t)) {
         timestamp_creation = std::chrono::duration<double>(
                std::chrono::system_clock::now().time_since_epoch()).count();
    }
};


// --- Core Interface Definition ---
// Modules will interact with the EANE core (Python or a C++ mock) through this interface.
class CoreInterface {
public:
    virtual ~CoreInterface() = default;

    // Global State Access
    virtual GlobalSelfStateCpp& get_global_state() = 0; // For read-write access
    virtual const GlobalSelfStateCpp& get_global_state_const() const = 0; // For read-only access

    // Event Queue Interaction
    virtual void event_queue_put(const EventDataCpp& event_data) = 0;
    // virtual bool event_queue_get_specific_cpp(const std::string& type_filter, EventDataCpp& out_event, double timeout_s) = 0;
    // virtual bool event_queue_get_specific_list_cpp(const std::vector<std::string>& type_filters, EventDataCpp& out_event, double timeout_s) = 0;

    // Time & Cycle Information
    virtual double get_current_timestamp() const = 0;
    virtual int get_current_cycle_num() const = 0;
    virtual double get_time_delta_continuous() const = 0;

    // Logging
    // Levels: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    virtual void log_message(const std::string& level, const std::string& module_name_cpp, const std::string& message) const = 0;

    // Module Access (Conceptual - direct C++ module access is complex across Python/C++ boundary)
    // This might involve Python calls if modules are primarily Python-managed.
    // virtual std::any get_module_interface_cpp(const std::string& module_name) = 0;

    // Shimyureshon Interface (Conceptual for C++)
    // These would likely trigger Python-side Shimyureshon logic.
    // virtual std::string request_shimyureshon_start_cpp(const std::string& sandbox_type, const std::map<std::string, std::any>& params) = 0;
    // virtual std::map<std::string, std::any> get_shimyureshon_results_cpp(const std::string& sh_id) = 0;
};
// eane_cpp_modules/mathematical_toolkit/mathematical_toolkit.h
#pragma once

#include "../core_interface.h" // For logging and potentially other core interactions
#include <Eigen/Dense>
#include <Eigen/Eigenvalues> // For SelfAdjointEigenSolver
#include <vector>
#include <string>
#include <functional>
#include <map>
#include <random>   // For std::mt19937, std::normal_distribution, etc.
#include <memory>   // For std::unique_ptr

// Forward declarations for simulators to avoid circular dependencies if they use MTK
class QuantumMechanicsSimulator;
class CosmologySimulator;
class StochasticSimulator;

class MathematicalToolkit {
public:
    // Constructor can optionally take a CoreInterface for logging or other purposes.
    explicit MathematicalToolkit(CoreInterface* core_ref = nullptr);
    ~MathematicalToolkit();

    // --- Linear Algebra ---
    // Solves Ax = b. Returns empty vector on failure.
    Eigen::VectorXd solve_linear_system(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) const;
    // Performs eigen decomposition for a self-adjoint (symmetric real) matrix.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_decomposition(const Eigen::MatrixXd& A_symmetric) const;

    // --- Calculus ---
    // Simple Runge-Kutta 4th order ODE solver.
    // func: dy/dt = f(t, y). y0 is the initial condition vector.
    // Returns a matrix where columns are solution vectors y at each t_eval point.
    Eigen::MatrixXd integrate_ode_rk4(
        const std::function<Eigen::VectorXd(double t, const Eigen::VectorXd& y)>& func,
        double t_start, double t_end, const Eigen::VectorXd& y0,
        int num_steps) const;
    
    // Numerical differentiation (central difference) - Placeholder
    double numerical_derivative_scalar_stub(const std::function<double(double)>& func, double x, double h = 1e-5) const;

    // --- Optimization (Stubs) ---
    // Placeholder for a simple scalar function minimizer using gradient descent.
    double minimize_scalar_function_gradient_descent_stub(
        const std::function<double(double)>& func,        // Function to minimize
        const std::function<double(double)>& grad_func,   // Gradient of the function
        double initial_guess,
        double learning_rate = 0.01,
        int max_iterations = 100,
        double tolerance = 1e-6) const;

    // --- Statistics ---
    // Calculates Shannon entropy for a discrete probability distribution.
    double calculate_shannon_entropy(const Eigen::VectorXd& probabilities) const;
    
    // Normal distribution functions
    double normal_pdf(double x, double mu, double sigma) const;
    double normal_cdf(double x, double mu, double sigma) const;
    Eigen::VectorXd generate_normal_samples(double mu, double sigma, int n_samples) const;

    // Uniform distribution functions
    Eigen::VectorXd generate_uniform_samples(double min_val, double max_val, int n_samples) const;

    // --- Access to Simulators ---
    // The MTK owns the simulator instances.
    QuantumMechanicsSimulator& qms();
    const QuantumMechanicsSimulator& qms() const;
    CosmologySimulator& csm();
    const CosmologySimulator& csm() const;
    StochasticSimulator& ssm();
    const StochasticSimulator& ssm() const;

    // --- Constants ---
    const std::map<std::string, double>& get_constants() const;

private:
    CoreInterface* core_ = nullptr; // Non-owning pointer to the core, for logging etc.
    std::map<std::string, double> constants_;
    
    // Simulators are owned by MTK
    std::unique_ptr<QuantumMechanicsSimulator> qms_ptr_;
    std::unique_ptr<CosmologySimulator> csm_ptr_;
    std::unique_ptr<StochasticSimulator> ssm_ptr_;

    // Mersenne Twister random number generator for internal statistical functions
    // Mutable to allow const methods like generate_normal_samples to use it.
    // Ensure thread safety if MTK methods are called concurrently (not an issue for single-threaded Python calls).
    mutable std::mt19937 rng_mtk_; 
};
// eane_cpp_modules/mathematical_toolkit/mathematical_toolkit.cpp
#include "mathematical_toolkit.h"
#include "../physics_simulators/physics_simulators.h" // Full include for instantiation
#include <cmath>     // For M_PI, M_E, std::exp, std::sqrt, std::erf, std::log2, std::pow
#include <numeric>   // For std::accumulate (not directly used but good for sums)
#include <algorithm> // For std::transform (not directly used but good for vector ops)
#include <stdexcept> // For std::runtime_error, std::invalid_argument

MathematicalToolkit::MathematicalToolkit(CoreInterface* core_ref)
    : core_(core_ref), rng_mtk_(std::random_device{}()) { // Seed with random_device
    constants_["pi"] = M_PI;
    constants_["e"] = M_E;
    constants_["c_mps"] = 299792458.0; // Speed of light in m/s
    constants_["hbar_Js"] = 1.054571817e-34; // Reduced Planck constant in J*s
    constants_["hbar_eVs"] = 6.582119569e-16; // Reduced Planck constant in eV*s
    constants_["kb_J_K"] = 1.380649e-23; // Boltzmann constant in J/K
    constants_["kb_eV_K"] = 8.617333262145e-5; // Boltzmann constant in eV/K
    constants_["amu_kg"] = 1.66053906660e-27; // Atomic mass unit in kg
    constants_["amu_MeV_c2"] = 931.49410242;   // Atomic mass unit in MeV/c^2
    constants_["electron_mass_kg"] = 9.1093837015e-31;
    constants_["electron_mass_eV_c2"] = 0.51099895000e6; // MeV/c^2 * 1e6

    // Instantiate simulators, passing reference to this MTK
    qms_ptr_ = std::make_unique<QuantumMechanicsSimulator>(*this);
    csm_ptr_ = std::make_unique<CosmologySimulator>(*this);
    ssm_ptr_ = std::make_unique<StochasticSimulator>(*this);

    if (core_) {
        core_->log_message("INFO", "MathematicalToolkitCpp", "MathematicalToolkit (C++) initialized successfully with integrated simulators.");
    }
    // No std::cout for production code, rely on core_ logger.
}

MathematicalToolkit::~MathematicalToolkit() {
    // unique_ptr will automatically clean up the simulators.
    if (core_) {
        core_->log_message("INFO", "MathematicalToolkitCpp", "MathematicalToolkit (C++) destroyed.");
    }
}

// --- Linear Algebra ---
Eigen::VectorXd MathematicalToolkit::solve_linear_system(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) const {
    if (A.rows() != A.cols() || A.rows() != b.size()) {
        if (core_) core_->log_message("ERROR", "MathematicalToolkitCpp", "SolveLinearSystem: Invalid matrix or vector dimensions. A: (" + std::to_string(A.rows()) + "," + std::to_string(A.cols()) + "), b: (" + std::to_string(b.size()) + ").");
        return Eigen::VectorXd(); // Return empty vector
    }
    if (A.rows() == 0) {
        if (core_) core_->log_message("WARNING", "MathematicalToolkitCpp", "SolveLinearSystem: Called with empty matrix A.");
        return Eigen::VectorXd();
    }
    // Using QR decomposition with column pivoting for robustness, good for potentially ill-conditioned systems.
    return A.colPivHouseholderQr().solve(b);
}

Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> MathematicalToolkit::eigen_decomposition(const Eigen::MatrixXd& A_symmetric) const {
    if (A_symmetric.rows() != A_symmetric.cols()) {
        if (core_) core_->log_message("ERROR", "MathematicalToolkitCpp", "EigenDecomposition: Matrix must be square.");
        // Fallback: return solver for a zero matrix of compatible (though incorrect) size or an empty one
        return Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(Eigen::MatrixXd::Zero(A_symmetric.rows(), A_symmetric.rows()));
    }
    if (A_symmetric.rows() == 0) {
         if (core_) core_->log_message("WARNING", "MathematicalToolkitCpp", "EigenDecomposition: Called with empty matrix.");
        return Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>();
    }
    // Note: SelfAdjointEigenSolver assumes the matrix IS self-adjoint.
    // No explicit check here, relies on caller providing a symmetric real matrix.
    return Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(A_symmetric, Eigen::ComputeEigenvectors);
}

// --- Calculus ---
Eigen::MatrixXd MathematicalToolkit::integrate_ode_rk4(
    const std::function<Eigen::VectorXd(double t, const Eigen::VectorXd& y)>& func,
    double t_start, double t_end, const Eigen::VectorXd& y0,
    int num_steps) const {
    if (num_steps <= 0) {
        if (core_) core_->log_message("ERROR", "MathematicalToolkitCpp", "IntegrateODE_RK4: num_steps must be positive.");
        return Eigen::MatrixXd(y0.size(), 0); // Return empty matrix
    }
    if (t_start == t_end) { // No integration interval
        Eigen::MatrixXd solution(y0.size(), 1);
        solution.col(0) = y0;
        return solution;
    }
    double dt = (t_end - t_start) / static_cast<double>(num_steps);
    Eigen::MatrixXd solution(y0.size(), num_steps + 1);
    solution.col(0) = y0;
    Eigen::VectorXd current_y = y0;
    double current_t = t_start;

    for (int i = 0; i < num_steps; ++i) {
        Eigen::VectorXd k1 = func(current_t, current_y);
        Eigen::VectorXd k2 = func(current_t + 0.5 * dt, current_y + 0.5 * dt * k1);
        Eigen::VectorXd k3 = func(current_t + 0.5 * dt, current_y + 0.5 * dt * k2);
        Eigen::VectorXd k4 = func(current_t + dt, current_y + dt * k3);

        current_y += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        current_t += dt;
        solution.col(i + 1) = current_y;
    }
    return solution;
}

double MathematicalToolkit::numerical_derivative_scalar_stub(const std::function<double(double)>& func, double x, double h) const {
    if (h == 0.0) {
        if (core_) core_->log_message("WARNING", "MathematicalToolkitCpp", "NumericalDerivative: h cannot be zero.");
        return 0.0; // Or NaN
    }
    return (func(x + h) - func(x - h)) / (2.0 * h);
}

double MathematicalToolkit::minimize_scalar_function_gradient_descent_stub(
    const std::function<double(double)>& func,
    const std::function<double(double)>& grad_func,
    double initial_guess,
    double learning_rate,
    int max_iterations,
    double tolerance) const {
    double x = initial_guess;
    for (int i = 0; i < max_iterations; ++i) {
        double grad = grad_func(x);
        double x_new = x - learning_rate * grad;
        if (std::abs(x_new - x) < tolerance || std::abs(grad) < tolerance) {
            break;
        }
        x = x_new;
    }
    return x;
}

// --- Statistics ---
double MathematicalToolkit::calculate_shannon_entropy(const Eigen::VectorXd& probabilities) const {
    if (probabilities.minCoeff() < -1e-9 || std::abs(probabilities.sum() - 1.0) > 1e-6) {
        if (core_) core_->log_message("WARNING", "MathematicalToolkitCpp", "CalculateEntropy: Invalid probability distribution (negative values or does not sum to 1).");
        // Return a value indicating error or handle as per system policy, e.g., max entropy.
        return probabilities.size() > 0 ? std::log2(static_cast<double>(probabilities.size())) : 0.0;
    }
    double entropy = 0.0;
    for (int i = 0; i < probabilities.size(); ++i) {
        double p = probabilities(i);
        if (p > 1e-12) { // Avoid log(0) issues
            entropy -= p * std::log2(p);
        }
    }
    return entropy;
}

double MathematicalToolkit::normal_pdf(double x, double mu, double sigma) const {
    if (sigma <= 1e-9) { // Avoid division by zero or near-zero sigma
        if (core_) core_->log_message("WARNING", "MathematicalToolkitCpp", "NormalPDF: Sigma is too small or non-positive.");
        return (std::abs(x - mu) < 1e-9) ? std::numeric_limits<double>::infinity() : 0.0; // Dirac delta conceptual
    }
    static const double inv_sqrt_2pi = 1.0 / std::sqrt(2.0 * M_PI);
    double z = (x - mu) / sigma;
    return (inv_sqrt_2pi / sigma) * std::exp(-0.5 * z * z);
}

double MathematicalToolkit::normal_cdf(double x, double mu, double sigma) const {
    if (sigma <= 1e-9) {
        if (core_) core_->log_message("WARNING", "MathematicalToolkitCpp", "NormalCDF: Sigma is too small or non-positive.");
        return (x < mu) ? 0.0 : 1.0;
    }
    return 0.5 * (1.0 + std::erf((x - mu) / (sigma * std::sqrt(2.0))));
}

Eigen::VectorXd MathematicalToolkit::generate_normal_samples(double mu, double sigma, int n_samples) const {
    if (n_samples <= 0) return Eigen::VectorXd();
    std::normal_distribution<double> dist(mu, sigma);
    Eigen::VectorXd samples(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        samples(i) = dist(rng_mtk_);
    }
    return samples;
}

Eigen::VectorXd MathematicalToolkit::generate_uniform_samples(double min_val, double max_val, int n_samples) const {
    if (n_samples <= 0) return Eigen::VectorXd();
    if (min_val >= max_val) {
        if (core_) core_->log_message("WARNING", "MathematicalToolkitCpp", "GenerateUniformSamples: min_val must be less than max_val.");
        // Return vector of min_val or handle error
        return Eigen::VectorXd::Constant(n_samples, min_val);
    }
    std::uniform_real_distribution<double> dist(min_val, max_val);
    Eigen::VectorXd samples(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        samples(i) = dist(rng_mtk_);
    }
    return samples;
}

// --- Access to Simulators ---
QuantumMechanicsSimulator& MathematicalToolkit::qms() { return *qms_ptr_; }
const QuantumMechanicsSimulator& MathematicalToolkit::qms() const { return *qms_ptr_; }
CosmologySimulator& MathematicalToolkit::csm() { return *csm_ptr_; }
const CosmologySimulator& MathematicalToolkit::csm() const { return *csm_ptr_; }
StochasticSimulator& MathematicalToolkit::ssm() { return *ssm_ptr_; }
const StochasticSimulator& MathematicalToolkit::ssm() const { return *ssm_ptr_; }

const std::map<std::string, double>& MathematicalToolkit::get_constants() const {
    return constants_;
}
// eane_cpp_modules/physics_simulators/physics_simulators.h
#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <functional>
#include <map>
#include <memory>   // For std::unique_ptr if owning internal resources

class MathematicalToolkit; // Forward declaration for MTK reference

class QuantumMechanicsSimulator {
public:
    explicit QuantumMechanicsSimulator(MathematicalToolkit& mtk_ref); // Requires a reference to MTK

    // Solves 1D time-independent Schrödinger equation using finite differences.
    // potential_func_eV: V(x) in eV, where x is in Angstroms.
    // particle_mass_amu: Mass of the particle in atomic mass units (amu).
    // Returns a map containing:
    //   "eigenvalues_eV": Eigen::VectorXd of energy eigenvalues in eV.
    //   "eigenstates_normalized": Eigen::MatrixXd where rows are normalized eigenvectors (wavefunctions).
    //   "x_grid_angstroms": Eigen::VectorXd of the x-coordinates used, in Angstroms.
    std::map<std::string, Eigen::MatrixXd> solve_schrodinger_1d_finite_diff(
        const std::function<double(double x_angstrom)>& potential_func_eV,
        double x_min_angstrom, double x_max_angstrom, int num_points,
        int num_eigenstates_to_return = 3,
        double particle_mass_amu = 1.007276); // Default to proton mass

private:
    MathematicalToolkit& mtk_; // Reference to the parent MTK for constants etc.
};

class CosmologySimulator {
public:
    explicit CosmologySimulator(MathematicalToolkit& mtk_ref);

    // Calculates scale factor a(t_lookback) for a flat LambdaCDM universe.
    // t_lookback_gyr: Lookback time from today in Gyr (positive for past, negative for future).
    // Returns Eigen::VectorXd of scale factors a(t), normalized so a(today) = 1.
    Eigen::VectorXd scale_factor_lcdm_flat(const Eigen::VectorXd& t_lookback_gyr_vec) const;

    // Calculates Hubble parameter H(z) in km/s/Mpc for a given redshift z.
    double hubble_parameter_at_z(double redshift) const;
    
    // Calculates comoving distance to redshift z in Mpc. (Conceptual, needs integration)
    // double comoving_distance_mpc(double redshift) const;

private:
    MathematicalToolkit& mtk_;
    // Cosmological parameters (can be made configurable)
    // Defaulting to values similar to Planck 2018 for consistency.
    double H0_km_s_Mpc_ = 67.36;    // Hubble constant today
    double Omega_m0_ = 0.3153;    // Total matter density parameter today
    double Omega_lambda0_ = 0.6847; // Dark energy density parameter today
    double Omega_r0_approx_ = 9.2e-5; // Radiation density (photons + relativistic neutrinos) - approximate
                                    // Omega_k0_ is 0 for a flat universe (1 - Omega_m0_ - Omega_lambda0_ - Omega_r0_)
};

class StochasticSimulator {
public:
    explicit StochasticSimulator(MathematicalToolkit& mtk_ref);

    // Generates event times for a homogeneous Poisson process.
    // mean_rate: Average number of events per unit time.
    // time_duration: Total duration to simulate.
    // Returns a sorted Eigen::VectorXd of event times.
    Eigen::VectorXd generate_poisson_process_events(double mean_rate, double time_duration) const;

    // Generates a 1D random walk.
    // num_steps: Number of steps in the walk.
    // step_size_std_dev: Standard deviation of the Gaussian step size.
    // initial_pos: Starting position of the walk.
    // Returns an Eigen::MatrixXd (1x(num_steps+1)) of positions at each step.
    Eigen::MatrixXd generate_random_walk_1d(int num_steps, double step_size_std_dev = 1.0, double initial_pos = 0.0) const;
    
    // (Future) Generate from other distributions (e.g., Exponential, Gamma)
    // (Future) Simulate Markov chains

private:
    MathematicalToolkit& mtk_;
    mutable std::mt19937 rng_ssm_; // Random number generator specific to this simulator
};
// eane_cpp_modules/physics_simulators/quantum_mechanics_simulator.cpp
#include "physics_simulators.h"
#include "../mathematical_toolkit/mathematical_toolkit.h" // For constants
#include <Eigen/Eigenvalues> // For SelfAdjointEigenSolver
#include <stdexcept>    // For runtime_error
#include <algorithm>    // For std::min

QuantumMechanicsSimulator::QuantumMechanicsSimulator(MathematicalToolkit& mtk_ref) : mtk_(mtk_ref) {
    if (mtk_.get_constants().find("hbar_eVs") == mtk_.get_constants().end() ||
        mtk_.get_constants().find("electron_mass_eV_c2") == mtk_.get_constants().end() || // Example mass
        mtk_.get_constants().find("c_mps") == mtk_.get_constants().end()) {
        // This check is more for robustness; MTK constructor should populate these.
        // In a real scenario, throw an error or log if critical constants are missing.
        // For now, assume MTK is correctly initialized.
    }
    // Log a message if core_ is available in mtk_
    // if (auto core_ptr = mtk_... ; core_ptr) { // How to get core from mtk? mtk needs a get_core method or similar
    //    core_ptr->log_message("INFO", "QuantumMechanicsSimulatorCpp", "QMS Initialized.");
    // }
}

std::map<std::string, Eigen::MatrixXd> QuantumMechanicsSimulator::solve_schrodinger_1d_finite_diff(
    const std::function<double(double x_angstrom)>& potential_func_eV,
    double x_min_angstrom, double x_max_angstrom, int num_points,
    int num_eigenstates_to_return,
    double particle_mass_amu) {

    if (num_points < 5) { // Need at least a few points for a meaningful discretization
        // mtk_.core_->log_message("ERROR", "QMS", "num_points too small for Schrödinger solver.");
        throw std::invalid_argument("QMS: num_points must be at least 5 for solve_schrodinger_1d_finite_diff.");
    }
    if (x_min_angstrom >= x_max_angstrom) {
        throw std::invalid_argument("QMS: x_min_angstrom must be less than x_max_angstrom.");
    }
    if (num_eigenstates_to_return <= 0) {
        throw std::invalid_argument("QMS: num_eigenstates_to_return must be positive.");
    }
    if (particle_mass_amu <= 0) {
        throw std::invalid_argument("QMS: particle_mass_amu must be positive.");
    }

    Eigen::VectorXd x_grid_angstroms = Eigen::VectorXd::LinSpaced(num_points, x_min_angstrom, x_max_angstrom);
    double dx_angstrom = (x_max_angstrom - x_min_angstrom) / static_cast<double>(num_points - 1);

    if (dx_angstrom < 1e-9) { // Avoid division by zero if dx is pathologically small
         throw std::runtime_error("QMS: dx_angstrom is too small, leading to numerical instability.");
    }

    // Constants from MTK
    const double hbar_eVs = mtk_.get_constants().at("hbar_eVs");
    const double c_mps = mtk_.get_constants().at("c_mps");
    const double amu_MeV_c2_val = mtk_.get_constants().at("amu_MeV_c2");
    double particle_mass_eV_c2 = particle_mass_amu * amu_MeV_c2_val * 1e6; // Convert MeV to eV

    // hbar * c in eV * Angstrom (approx. 1973.27 eV * Angstrom)
    double hbar_c_eV_Angstrom = hbar_eVs * c_mps * 1e10; // 1m = 1e10 Angstrom

    // Kinetic energy factor: (hbar*c)^2 / (2 * m*c^2) in units of eV * Angstrom^2
    double kinetic_factor_eV_Angstrom_sq = std::pow(hbar_c_eV_Angstrom, 2) / (2.0 * particle_mass_eV_c2);

    // Coefficient for the T operator in Hamiltonian: kinetic_factor_eV_Angstrom_sq / (dx_angstrom^2)
    double T_coeff_eV = kinetic_factor_eV_Angstrom_sq / (dx_angstrom * dx_angstrom);

    Eigen::MatrixXd H_matrix = Eigen::MatrixXd::Zero(num_points, num_points);

    // Populate Hamiltonian matrix (Tridiagonal for 1D finite difference method)
    for (int i = 0; i < num_points; ++i) {
        // Diagonal elements: 2*T_coeff_eV + V(x_i)
        H_matrix(i, i) = 2.0 * T_coeff_eV + potential_func_eV(x_grid_angstroms(i));
        // Off-diagonal elements: -T_coeff_eV
        if (i > 0) {
            H_matrix(i, i - 1) = -T_coeff_eV;
        }
        if (i < num_points - 1) {
            H_matrix(i, i + 1) = -T_coeff_eV;
        }
    }

    // Solve the eigenvalue problem H * Psi = E * Psi
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(H_matrix);
    if (eigensolver.info() != Eigen::Success) {
        // mtk_.core_->log_message("ERROR", "QMS", "Eigenvalue decomposition failed in Schrödinger solver.");
        throw std::runtime_error("QMS: Eigenvalue decomposition failed.");
    }

    Eigen::VectorXd all_eigenvalues_eV = eigensolver.eigenvalues();
    Eigen::MatrixXd all_eigenvectors = eigensolver.eigenvectors(); // Columns are eigenvectors

    int num_solutions_found = static_cast<int>(all_eigenvalues_eV.size());
    int num_to_return = std::min(num_eigenstates_to_return, num_solutions_found);

    Eigen::VectorXd selected_eigenvalues = all_eigenvalues_eV.head(num_to_return);
    Eigen::MatrixXd selected_eigenstates_cols = all_eigenvectors.leftCols(num_to_return);

    // Normalize eigenvectors (wavefunctions): integral |Psi(x)|^2 dx = 1
    // Psi_discrete_normalized = Psi_discrete / sqrt( sum(Psi_discrete_i^2 * dx) )
    Eigen::MatrixXd normalized_eigenstates_rows(num_to_return, num_points);
    for (int k = 0; k < num_to_return; ++k) {
        Eigen::VectorXd psi_k = selected_eigenstates_cols.col(k);
        double norm_sq_integral = psi_k.squaredNorm() * dx_angstrom; // Approximate integral
        if (norm_sq_integral > 1e-12) {
            normalized_eigenstates_rows.row(k) = psi_k.transpose() / std::sqrt(norm_sq_integral);
        } else {
            // Handle case of zero norm (e.g., if Psi_k is all zeros, though unlikely for physical solutions)
            normalized_eigenstates_rows.row(k) = psi_k.transpose(); // Keep as is or zero out
        }
    }

    std::map<std::string, Eigen::MatrixXd> result;
    result["eigenvalues_eV"] = selected_eigenvalues;
    result["eigenstates_normalized"] = normalized_eigenstates_rows; // Store as rows for convention
    result["x_grid_angstroms"] = x_grid_angstroms;

    return result;
}
// eane_cpp_modules/physics_simulators/cosmology_simulator.cpp
#include "physics_simulators.h"
#include "../mathematical_toolkit/mathematical_toolkit.h"
#include <cmath>     // std::pow, std::sqrt, std::sinh, std::exp
#include <stdexcept> // For runtime_error
#include <algorithm> // For std::clamp

CosmologySimulator::CosmologySimulator(MathematicalToolkit& mtk_ref) : mtk_(mtk_ref) {
    // Verify that necessary constants are available in MTK
    if (mtk_.get_constants().find("c_mps") == mtk_.get_constants().end() ||
        // H0, Omegas are member variables now, not from MTK constants directly
        H0_km_s_Mpc_ <= 0 || Omega_m0_ < 0 || Omega_lambda0_ < 0) {
        // This indicates a setup issue.
        // Log if mtk has core access
    }
}

Eigen::VectorXd CosmologySimulator::scale_factor_lcdm_flat(const Eigen::VectorXd& t_lookback_gyr_vec) const {
    // Physical constants and conversions
    const double GYR_TO_SECONDS = 3.15576e16; // Seconds in a Gyr
    const double MPC_TO_KM = 3.08567758e19;    // Kilometers in a Megaparsec

    // Hubble parameter in 1/Gyr
    double H0_per_Gyr = H0_km_s_Mpc_ * (GYR_TO_SECONDS / MPC_TO_KM);
    double t_age_universe_approx_gyr = 1.0 / H0_per_Gyr; // Hubble time, rough estimate of age
    // A more common estimate for Planck 2018 is ~13.787 Gyr
    t_age_universe_approx_gyr = 13.787;


    Eigen::VectorXd scale_factors(t_lookback_gyr_vec.size());

    for (int i = 0; i < t_lookback_gyr_vec.size(); ++i) {
        double t_lookback_gyr = t_lookback_gyr_vec(i);
        double t_from_bb_gyr = t_age_universe_approx_gyr - t_lookback_gyr;

        if (t_from_bb_gyr <= 1e-6) { // Very early universe, close to t=0
            // Approximation for matter-dominated era: a(t) ~ t^(2/3)
            // (or radiation-dominated a(t) ~ t^(1/2) if even earlier)
            // Normalized to a very small value.
            scale_factors(i) = 1e-9; // Avoid singularity
        } else if (t_lookback_gyr < -50.0) { // Far future (t_from_bb_gyr > age + 50 Gyr)
            // Exponential growth dominated by Lambda
            double t_future_from_now_gyr = -t_lookback_gyr;
            scale_factors(i) = std::exp(std::sqrt(Omega_lambda0_) * H0_per_Gyr * t_future_from_now_gyr);
        }
        else {
            // Using an approximate analytical solution for flat LambdaCDM (valid for t > 0)
            // a(t) = (Omega_m0 / Omega_lambda0)^(1/3) * [sinh( (3/2) * sqrt(Omega_lambda0) * H0 * t )]^(2/3)
            // This needs to be normalized such that a(t_age_universe_approx_gyr) = 1.
            
            double sinh_arg = 1.5 * std::sqrt(Omega_lambda0_) * H0_per_Gyr * t_from_bb_gyr;
            double sinh_term = std::sinh(sinh_arg);

            if (sinh_term < 0 && sinh_arg > 0) sinh_term = 0; // sinh(small_positive) shouldn't be negative
            else if (sinh_term > 0 && sinh_arg < 0) sinh_term = 0; // sinh(small_negative)

            double term1_factor = Omega_m0_ / Omega_lambda0_;
            
            double a_t_unnormalized;
            if (term1_factor <= 0 || sinh_term <= 0) { // Fallback for early times or problematic parameters
                // Matter-dominated approx: a(t) ~ ( (3/2) * sqrt(Omega_m0) * H0 * t )^(2/3)
                // This is problematic if Omega_m0 is also very small.
                // Simplest fallback: linear with small power for early times
                a_t_unnormalized = std::pow(H0_per_Gyr * t_from_bb_gyr, 0.5); // More like radiation era
            } else {
                 a_t_unnormalized = std::pow(term1_factor, 1.0/3.0) * std::pow(sinh_term, 2.0/3.0);
            }


            // Normalization factor: a_today_unnormalized
            double sinh_arg_today = 1.5 * std::sqrt(Omega_lambda0_) * H0_per_Gyr * t_age_universe_approx_gyr;
            double a_today_unnormalized = std::pow(term1_factor, 1.0/3.0) * std::pow(std::sinh(sinh_arg_today), 2.0/3.0);

            if (std::abs(a_today_unnormalized) > 1e-9) {
                scale_factors(i) = a_t_unnormalized / a_today_unnormalized;
            } else {
                // This case implies issues with parameters or very early time calc for a_today
                scale_factors(i) = (t_from_bb_gyr / t_age_universe_approx_gyr); // Simple linear scaling as last resort
            }
        }
        scale_factors(i) = std::clamp(scale_factors(i), 1e-9, 1000.0); // Clamp to avoid extreme values
    }
    return scale_factors;
}

double CosmologySimulator::hubble_parameter_at_z(double redshift) const {
    if (redshift < -1.0 + 1e-9) { // (1+z) must be > 0
        // Handle error: redshift implies scale factor <= 0
        // For very negative redshift (far future), (1+z) can be very small.
        // mtk_.core_->log_message("WARNING", "CSM", "Redshift < -1 is unphysical for H(z) formula.");
        // This implies an issue with how redshift is being used if it's for past times.
        // If it's for future (negative z), |z| < 1 for typical models before a turnaround.
        return 0.0; // Or throw
    }
    double one_plus_z = 1.0 + redshift;
    double term_m = Omega_m0_ * std::pow(one_plus_z, 3);
    double term_r = Omega_r0_approx_ * std::pow(one_plus_z, 4); // Radiation
    double term_lambda = Omega_lambda0_;
    // For a flat universe, Omega_k0 = 0. The curvature term would be Omega_k0 * (1+z)^2.
    // Total density parameter: Omega_total0 = Omega_m0_ + Omega_lambda0_ + Omega_r0_approx_
    // Curvature: Omega_k0 = 1.0 - Omega_total0
    // For simplicity, assume flat as per typical LambdaCDM.
    
    double E_z_squared = term_m + term_r + term_lambda;
    if (E_z_squared < 0) {
        // This shouldn't happen for standard cosmological parameters and z > -1.
        // mtk_.core_->log_message("WARNING", "CSM", "E(z)^2 is negative in H(z) calculation.");
        return H0_km_s_Mpc_; // Fallback or error
    }
    return H0_km_s_Mpc_ * std::sqrt(E_z_squared);
}
// eane_cpp_modules/physics_simulators/stochastic_simulator.cpp
#include "physics_simulators.h"
#include "../mathematical_toolkit/mathematical_toolkit.h" // Not strictly needed for these stubs if MTK doesn't provide dists
#include <random>    // For std::poisson_distribution, std::normal_distribution, std::uniform_real_distribution
#include <algorithm> // For std::sort
#include <vector>    // For internal storage before Eigen conversion
#include <stdexcept> // For std::invalid_argument

StochasticSimulator::StochasticSimulator(MathematicalToolkit& mtk_ref)
    : mtk_(mtk_ref), rng_ssm_(std::random_device{}()) {
    // Log if mtk provides core access
}

Eigen::VectorXd StochasticSimulator::generate_poisson_process_events(double mean_rate, double time_duration) const {
    if (mean_rate < 0) {
        throw std::invalid_argument("SSM: Poisson process mean_rate must be non-negative.");
    }
    if (time_duration < 0) {
        throw std::invalid_argument("SSM: Poisson process time_duration must be non-negative.");
    }
    if (mean_rate == 0 || time_duration == 0) {
        return Eigen::VectorXd(0); // No events
    }

    std::poisson_distribution<int> num_events_dist(mean_rate * time_duration);
    int num_events = num_events_dist(rng_ssm_);

    if (num_events <= 0) {
        return Eigen::VectorXd(0);
    }

    std::uniform_real_distribution<double> event_time_dist(0.0, time_duration);
    std::vector<double> event_times_vec(num_events);
    for (int i = 0; i < num_events; ++i) {
        event_times_vec[i] = event_time_dist(rng_ssm_);
    }
    std::sort(event_times_vec.begin(), event_times_vec.end());

    // Convert std::vector<double> to Eigen::VectorXd
    Eigen::Map<Eigen::VectorXd> eigen_event_times(event_times_vec.data(), num_events);
    return eigen_event_times;
}

Eigen::MatrixXd StochasticSimulator::generate_random_walk_1d(int num_steps, double step_size_std_dev, double initial_pos) const {
    if (num_steps < 0) {
        throw std::invalid_argument("SSM: Number of steps for random walk must be non-negative.");
    }
    if (step_size_std_dev < 0) {
        throw std::invalid_argument("SSM: Step size standard deviation must be non-negative.");
    }
    
    if (num_steps == 0) {
        Eigen::MatrixXd walk(1, 1);
        walk(0,0) = initial_pos;
        return walk;
    }


    Eigen::MatrixXd walk_positions(1, num_steps + 1);
    walk_positions(0, 0) = initial_pos;
    
    if (step_size_std_dev == 0.0) { // Deterministic walk (no change)
        for (int i = 1; i <= num_steps; ++i) {
            walk_positions(0, i) = initial_pos;
        }
        return walk_positions;
    }

    std::normal_distribution<double> step_dist(0.0, step_size_std_dev);
    for (int i = 0; i < num_steps; ++i) {
        walk_positions(0, i + 1) = walk_positions(0, i) + step_dist(rng_ssm_);
    }
    return walk_positions;
}
// eane_cpp_modules/subconscious_mind/subconscious_mind.h
#pragma once

#include "../core_interface.h" // For CoreInterface and GlobalSelfStateCpp
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <random>   // For std::mt19937
#include <map>      // For snapshot data (if returning map)

// Forward declaration
class CoreInterface;

class SubconsciousMind {
public:
    // Constructor
    SubconsciousMind(CoreInterface* core, 
                     int state_dim = 10, 
                     int output_dim_for_consciousness = 10,
                     int num_observation_features = 10); // Added num_observation_features

    // Main logic update, called by the core or scheduler
    void update_logic();

    // --- Snapshot and Restoration ---
    struct SnapshotDataCpp {
        Eigen::MatrixXd transition_matrix;
        Eigen::MatrixXd emission_matrix;
        Eigen::MatrixXd Wh_matrix; // Renamed from Wh_
        Eigen::VectorXd hidden_state_vec; // Renamed from hidden_state_
    };
    SnapshotDataCpp get_snapshot_data() const;
    bool restore_from_snapshot_data(const SnapshotDataCpp& data);

    // --- Accessors for state inspection or use by other C++ modules/Python ---
    Eigen::VectorXd get_current_influence_output() const;
    double get_current_influence_norm() const;
    Eigen::VectorXd get_hidden_state_copy() const;

private:
    CoreInterface* core_recombinator_; // Non-owning pointer to the core
    int state_dim_;
    int output_dim_for_consciousness_;
    int num_observation_features_; // Number of features in the observation vector

    Eigen::MatrixXd transition_matrix_;
    Eigen::MatrixXd emission_matrix_; // Maps observation to hidden state update component
    Eigen::MatrixXd Wh_matrix_;       // Maps hidden state to influence output
    Eigen::VectorXd hidden_state_vec_;

    // Internal state (mirroring Python module_state relevant parts)
    double current_influence_norm_ = 0.0;
    Eigen::VectorXd current_influence_output_for_consciousness_; // Cached output

    // Helper methods for internal calculations
    Eigen::VectorXd update_hidden_state_internal(const Eigen::VectorXd& observation_vector);
    Eigen::VectorXd compute_influence_internal();

    // Random number generator for any stochasticity (if needed in future, e.g. noise)
    // mutable std::mt19937 rng_scm_; // Mutable if used in const methods
};
// eane_cpp_modules/subconscious_mind/subconscious_mind.cpp
#include "subconscious_mind.h"
#include <cmath>     // For std::tanh, std::exp, std::clamp
#include <iostream>  // For logging if core_ is null during init

SubconsciousMind::SubconsciousMind(CoreInterface* core, 
                                   int state_dim, 
                                   int output_dim_for_consciousness,
                                   int num_observation_features)
    : core_recombinator_(core),
      state_dim_(state_dim),
      output_dim_for_consciousness_(output_dim_for_consciousness),
      num_observation_features_(num_observation_features)
      // rng_scm_(std::random_device{}()) // Initialize RNG if used
       {

    // Initialize matrices and vectors with appropriate sizes and random values
    if (state_dim_ > 0) {
        transition_matrix_ = Eigen::MatrixXd::Random(state_dim_, state_dim_) * 0.1;
        if (num_observation_features_ > 0) {
            emission_matrix_ = Eigen::MatrixXd::Random(state_dim_, num_observation_features_) * 0.1;
        } else {
            emission_matrix_.resize(state_dim_, 0); // Or handle error
        }
        hidden_state_vec_ = Eigen::VectorXd::Random(state_dim_) * 0.1;

        if (output_dim_for_consciousness_ > 0) {
            Wh_matrix_ = Eigen::MatrixXd::Random(output_dim_for_consciousness_, state_dim_) * 0.1;
            current_influence_output_for_consciousness_ = Eigen::VectorXd::Zero(output_dim_for_consciousness_);
        } else {
            Wh_matrix_.resize(0, state_dim_); // Or 0,0 if state_dim is also 0
            current_influence_output_for_consciousness_.resize(0);
        }
    } else { // state_dim_ <= 0 implies non-functional module
        state_dim_ = 0; // Ensure it's 0
        output_dim_for_consciousness_ = 0;
        num_observation_features_ = 0;
        transition_matrix_.resize(0,0);
        emission_matrix_.resize(0,0);
        hidden_state_vec_.resize(0);
        Wh_matrix_.resize(0,0);
        current_influence_output_for_consciousness_.resize(0);
    }

    if(core_recombinator_) {
        core_recombinator_->log_message("INFO", "SubconsciousMindCpp", 
            "SubconsciousMind C++ module initialized. State Dim: " + std::to_string(state_dim_) + 
            ", Output Dim: " + std::to_string(output_dim_for_consciousness_) +
            ", Obs Features: " + std::to_string(num_observation_features_));
    } else {
        // Fallback logging if core is not available during construction
        // std::cout << "SubconsciousMind C++ initialized (no core_ref)." << std::endl;
    }
}

void SubconsciousMind::update_logic() {
    if (!core_recombinator_ || state_dim_ <= 0 || num_observation_features_ <= 0) {
        // Module is not properly initialized or core is missing, do nothing.
        return;
    }

    GlobalSelfStateCpp& gs = core_recombinator_->get_global_state();

    // Construct the observation vector from GlobalSelfStateCpp
    // This must match the `num_observation_features_` dimension.
    Eigen::VectorXd observation_vector(num_observation_features_);
    // Ensure gs.needs has at least 3 elements before accessing, or provide defaults.
    double mean_needs = 0.5;
    if (gs.needs_vector.size() >= 3) { // Assuming 3 needs: Autonomy, Relatedness, Competence
        mean_needs = gs.needs_vector.mean();
    } else if (gs.needs_vector.size() > 0) {
        mean_needs = gs.needs_vector.mean(); // If fewer than 3, take mean of what's there
    }
    // Make sure we have 10 features as per Python's observation_components
    // The order must be consistent.
    if (num_observation_features_ == 10) {
        observation_vector << gs.valencia, gs.arousal, gs.motivacion, gs.dolor, 
                              gs.coherence_score, gs.system_entropy, gs.self_esteem,
                              mean_needs, gs.phi_consciousness, gs.resilience_stability_metric;
    } else {
        // Fallback or error if num_observation_features_ is not 10.
        // For now, fill with zeros or log error.
        if (core_recombinator_) core_recombinator_->log_message("ERROR", "SubconsciousMindCpp", "num_observation_features_ mismatch. Expected 10, got " + std::to_string(num_observation_features_));
        observation_vector = Eigen::VectorXd::Zero(num_observation_features_); // Default to zero vector to prevent crash
    }


    hidden_state_vec_ = update_hidden_state_internal(observation_vector);

    if (output_dim_for_consciousness_ > 0) {
        current_influence_output_for_consciousness_ = compute_influence_internal();
        current_influence_norm_ = current_influence_output_for_consciousness_.norm();
    } else {
        current_influence_norm_ = 0.0;
        // current_influence_output_for_consciousness_ is already size 0 if output_dim is 0
    }

    // Send an event with the updated influence (conceptual, depends on EventDataCpp definition)
    EventDataCpp influence_event("subconscious_influence_update_cpp");
    influence_event.content["influence_norm_val"] = current_influence_norm_; // Use _val to avoid pybind conflicts
    if (output_dim_for_consciousness_ > 0) {
         // Storing Eigen::VectorXd directly in EventContentValueCpp (if variant supports it)
        influence_event.content["influence_vector_val"] = current_influence_output_for_consciousness_;
    }
    influence_event.priority_label = "low";
    influence_event.source_module = "SubconsciousMindCpp";
    core_recombinator_->event_queue_put(influence_event);
}

Eigen::VectorXd SubconsciousMind::update_hidden_state_internal(const Eigen::VectorXd& observation_vector) {
    // emission_matrix_ is (state_dim_ x num_observation_features_)
    // observation_vector is (num_observation_features_ x 1)
    // prob_input should be (state_dim_ x 1)
    if (emission_matrix_.cols() != observation_vector.size()) {
        if (core_recombinator_) core_recombinator_->log_message("ERROR", "SubconsciousMindCpp", "Dimension mismatch: emission_matrix_.cols() != observation_vector.size()");
        return hidden_state_vec_; // Return current state to avoid crash
    }
    Eigen::VectorXd prob_input = emission_matrix_ * observation_vector;

    // Apply exp and normalize (as per Python logic, which is more like a weighted sum after exp)
    Eigen::VectorXd prob_exp = prob_input.array().unaryExpr([](double x_val){ 
        return std::exp(std::clamp(x_val, -100.0, 100.0)); // Clamp to avoid overflow
    });

    double prob_sum = prob_exp.sum();
    Eigen::VectorXd prob_norm_factor; // Renamed to avoid conflict
    if (prob_sum > 1e-9) {
        prob_norm_factor = prob_exp / prob_sum;
    } else {
        // Avoid division by zero; distribute equally if sum is too small
        if (prob_exp.size() > 0) {
            prob_norm_factor = Eigen::VectorXd::Constant(prob_exp.size(), 1.0 / static_cast<double>(prob_exp.size()));
        } else {
            prob_norm_factor.resize(0); // Or handle as error
        }
    }

    // transition_matrix_ is (state_dim_ x state_dim_)
    // hidden_state_vec_ is (state_dim_ x 1)
    // new_hidden_state_candidate should be (state_dim_ x 1)
    if (transition_matrix_.cols() != hidden_state_vec_.size() || prob_norm_factor.size() != hidden_state_vec_.size()) {
         if (core_recombinator_) core_recombinator_->log_message("ERROR", "SubconsciousMindCpp", "Dimension mismatch in hidden state update calculation.");
        return hidden_state_vec_;
    }
    Eigen::VectorXd new_hidden_state_candidate = transition_matrix_ * hidden_state_vec_ + prob_norm_factor;
    
    // Apply tanh activation
    return new_hidden_state_candidate.array().tanh();
}

Eigen::VectorXd SubconsciousMind::compute_influence_internal() {
    // Wh_matrix_ is (output_dim_for_consciousness_ x state_dim_)
    // hidden_state_vec_ is (state_dim_ x 1)
    // influence should be (output_dim_for_consciousness_ x 1)
    if (Wh_matrix_.cols() != hidden_state_vec_.size()) {
        if (core_recombinator_) core_recombinator_->log_message("ERROR", "SubconsciousMindCpp", "Dimension mismatch: Wh_matrix_.cols() != hidden_state_vec_.size()");
        return Eigen::VectorXd::Zero(output_dim_for_consciousness_); // Return zero vector
    }
    Eigen::VectorXd influence = Wh_matrix_ * hidden_state_vec_;
    
    // Apply tanh activation
    return influence.array().tanh();
}

// --- Accessors ---
Eigen::VectorXd SubconsciousMind::get_current_influence_output() const {
    return current_influence_output_for_consciousness_;
}

double SubconsciousMind::get_current_influence_norm() const {
    return current_influence_norm_;
}

Eigen::VectorXd SubconsciousMind::get_hidden_state_copy() const {
    return hidden_state_vec_; // Eigen vectors/matrices are copy-on-write or copied by value here
}

// --- Snapshot and Restoration ---
SubconsciousMind::SnapshotDataCpp SubconsciousMind::get_snapshot_data() const {
    return {transition_matrix_, emission_matrix_, Wh_matrix_, hidden_state_vec_};
}

bool SubconsciousMind::restore_from_snapshot_data(const SnapshotDataCpp& data) {
    // Perform dimension checks before assigning
    if (data.transition_matrix.rows() == state_dim_ && data.transition_matrix.cols() == state_dim_ &&
        data.emission_matrix.rows() == state_dim_ && data.emission_matrix.cols() == num_observation_features_ &&
        data.Wh_matrix.rows() == output_dim_for_consciousness_ && data.Wh_matrix.cols() == state_dim_ &&
        data.hidden_state_vec.size() == state_dim_) {
        
        transition_matrix_ = data.transition_matrix;
        emission_matrix_ = data.emission_matrix;
        Wh_matrix_ = data.Wh_matrix;
        hidden_state_vec_ = data.hidden_state_vec;

        // Reset runtime derived state
        if (output_dim_for_consciousness_ > 0) {
            current_influence_output_for_consciousness_ = Eigen::VectorXd::Zero(output_dim_for_consciousness_);
        } else {
            current_influence_output_for_consciousness_.resize(0);
        }
        current_influence_norm_ = 0.0; // Will be recalculated on next update_logic()

        if(core_recombinator_) core_recombinator_->log_message("INFO", "SubconsciousMindCpp", "State restored successfully from snapshot data.");
        return true;
    } else {
        if(core_recombinator_) core_recombinator_->log_message("ERROR", "SubconsciousMindCpp", "Snapshot data dimension mismatch. Restoration failed.");
        return false;
    }
}
// eane_cpp_modules/learning_module/lstm_stub.h
#pragma once
#include <Eigen/Dense>
#include <vector>

// Simplified LSTM cell structure for stub purposes
struct LSTMCellStateStub {
    Eigen::VectorXd hidden_state; // h_t
    Eigen::VectorXd cell_state;   // c_t
    LSTMCellStateStub(int hidden_dim = 0) : hidden_state(Eigen::VectorXd::Zero(hidden_dim)), cell_state(Eigen::VectorXd::Zero(hidden_dim)) {}
};

class LSTMStub {
public:
    LSTMStub(int input_dim, int hidden_dim, int output_dim);

    // Process a single time step
    Eigen::VectorXd step(const Eigen::VectorXd& input_vec, LSTMCellStateStub& prev_state);
    
    // Process a sequence of inputs
    // Returns a matrix where each row is the output_vec for that time step
    Eigen::MatrixXd process_sequence(const Eigen::MatrixXd& input_sequence); // input_sequence: rows=timesteps, cols=input_dim

    // Conceptual training step for a sequence
    // error_signals_sequence: rows=timesteps, cols=output_dim
    void train_sequence(const Eigen::MatrixXd& input_sequence, 
                        const Eigen::MatrixXd& target_output_sequence, 
                        double learning_rate);

    double get_last_loss() const; // Conceptual

private:
    int input_dim_;
    int hidden_dim_;
    int output_dim_;

    // LSTM weights and biases (simplified)
    Eigen::MatrixXd W_f_, W_i_, W_c_, W_o_; // Input weights
    Eigen::MatrixXd U_f_, U_i_, U_c_, U_o_; // Recurrent weights
    Eigen::VectorXd b_f_, b_i_, b_c_, b_o_; // Biases

    Eigen::MatrixXd W_y_; // Output layer weights
    Eigen::VectorXd b_y_; // Output layer bias

    LSTMCellStateStub current_cell_state_; // Stores h_t and c_t for next step
    double last_calculated_loss_ = 1.0; // Conceptual

    // Helper for activation functions
    static Eigen::VectorXd sigmoid(const Eigen::VectorXd& x);
    static Eigen::VectorXd tanh_act(const Eigen::VectorXd& x); // Renamed to avoid conflict
};
// eane_cpp_modules/learning_module/lstm_stub.cpp
#include "lstm_stub.h"
#include <cmath>     // std::exp, std::tanh
#include <iostream>  // For simple logging if needed

LSTMStub::LSTMStub(int input_dim, int hidden_dim, int output_dim)
    : input_dim_(input_dim), hidden_dim_(hidden_dim), output_dim_(output_dim),
      current_cell_state_(hidden_dim) {

    if (input_dim <= 0 || hidden_dim <= 0 || output_dim <= 0) {
        // Handle invalid dimensions, perhaps by setting flags or logging
        // For a stub, we might allow it but operations will be no-ops or return zeros.
        // std::cerr << "LSTMStub Warning: Invalid dimensions provided." << std::endl;
        // Ensure matrices are empty if dimensions are invalid to prevent Eigen errors
        input_dim_ = std::max(0, input_dim);
        hidden_dim_ = std::max(0, hidden_dim);
        output_dim_ = std::max(0, output_dim);
    }
    
    // Initialize weights with small random values if dimensions are valid
    if (input_dim_ > 0 && hidden_dim_ > 0) {
        auto init_random_matrix = [&](int rows, int cols){ return Eigen::MatrixXd::Random(rows, cols) * 0.1; };
        auto init_random_vector = [&](int size){ return Eigen::VectorXd::Random(size) * 0.1; };

        W_f_ = init_random_matrix(hidden_dim_, input_dim_); U_f_ = init_random_matrix(hidden_dim_, hidden_dim_); b_f_ = init_random_vector(hidden_dim_);
        W_i_ = init_random_matrix(hidden_dim_, input_dim_); U_i_ = init_random_matrix(hidden_dim_, hidden_dim_); b_i_ = init_random_vector(hidden_dim_);
        W_c_ = init_random_matrix(hidden_dim_, input_dim_); U_c_ = init_random_matrix(hidden_dim_, hidden_dim_); b_c_ = init_random_vector(hidden_dim_);
        W_o_ = init_random_matrix(hidden_dim_, input_dim_); U_o_ = init_random_matrix(hidden_dim_, hidden_dim_); b_o_ = init_random_vector(hidden_dim_);
    }
     if (hidden_dim_ > 0 && output_dim_ > 0) {
        W_y_ = Eigen::MatrixXd::Random(output_dim_, hidden_dim_) * 0.1;
        b_y_ = Eigen::VectorXd::Random(output_dim_) * 0.1;
    }
    // Initialize current_cell_state_ h and c to zeros
    current_cell_state_.hidden_state = Eigen::VectorXd::Zero(hidden_dim_);
    current_cell_state_.cell_state = Eigen::VectorXd::Zero(hidden_dim_);
}

Eigen::VectorXd LSTMStub::sigmoid(const Eigen::VectorXd& x) {
    return x.unaryExpr([](double val){ return 1.0 / (1.0 + std::exp(-val)); });
}

Eigen::VectorXd LSTMStub::tanh_act(const Eigen::VectorXd& x) {
    return x.array().tanh(); // Use Eigen's built-in tanh
}

Eigen::VectorXd LSTMStub::step(const Eigen::VectorXd& input_vec, LSTMCellStateStub& prev_state) {
    if (input_dim_ <= 0 || hidden_dim_ <= 0 || output_dim_ <= 0 || 
        input_vec.size() != input_dim_ || 
        prev_state.hidden_state.size() != hidden_dim_ || 
        prev_state.cell_state.size() != hidden_dim_ ) {
        // Return zero vector if dimensions are inconsistent or module not properly initialized
        return Eigen::VectorXd::Zero(output_dim_ > 0 ? output_dim_ : 0);
    }

    // Concatenate (conceptually) or use separate weights for input and hidden state
    // h_prev = prev_state.hidden_state
    // c_prev = prev_state.cell_state
    // x_t = input_vec

    // Gates
    Eigen::VectorXd f_t = sigmoid(W_f_ * input_vec + U_f_ * prev_state.hidden_state + b_f_); // Forget gate
    Eigen::VectorXd i_t = sigmoid(W_i_ * input_vec + U_i_ * prev_state.hidden_state + b_i_); // Input gate
    Eigen::VectorXd c_tilde_t = tanh_act(W_c_ * input_vec + U_c_ * prev_state.hidden_state + b_c_); // Candidate cell state
    Eigen::VectorXd o_t = sigmoid(W_o_ * input_vec + U_o_ * prev_state.hidden_state + b_o_); // Output gate

    // New cell state and hidden state
    current_cell_state_.cell_state = (f_t.array() * prev_state.cell_state.array()).matrix() + (i_t.array() * c_tilde_t.array()).matrix();
    current_cell_state_.hidden_state = (o_t.array() * tanh_act(current_cell_state_.cell_state).array()).matrix();
    
    // Update prev_state for the caller if it's meant to be an in-out parameter for external state management
    // Or, if this step updates the internal current_cell_state_, then process_sequence uses that.
    // For this stub, let's assume `step` updates `this->current_cell_state_`.
    // The `prev_state` argument could be used if an external entity manages the sequence's state.
    // For simplicity now, we'll primarily use the internal `current_cell_state_`.
    // The `prev_state` argument here would be more for passing in an initial state if needed.

    // Output
    Eigen::VectorXd output_vec = W_y_ * current_cell_state_.hidden_state + b_y_;
    // Optionally apply an output activation function here (e.g., linear, sigmoid, softmax)
    // For a generic stub, linear output is fine.

    return output_vec;
}

Eigen::MatrixXd LSTMStub::process_sequence(const Eigen::MatrixXd& input_sequence) {
    if (input_dim_ <= 0 || hidden_dim_ <= 0 || output_dim_ <= 0 || 
        input_sequence.cols() != input_dim_ || input_sequence.rows() == 0) {
        return Eigen::MatrixXd(0, output_dim_ > 0 ? output_dim_ : 0);
    }

    int seq_len = static_cast<int>(input_sequence.rows());
    Eigen::MatrixXd output_sequence(seq_len, output_dim_);

    // Reset internal state at the beginning of a new sequence (or take initial state as param)
    current_cell_state_.hidden_state.setZero();
    current_cell_state_.cell_state.setZero();

    for (int t = 0; t < seq_len; ++t) {
        Eigen::VectorXd input_t = input_sequence.row(t);
        // The `step` function uses and updates `this->current_cell_state_`
        // So, the prev_state passed to step should be a copy of the state *before* this step.
        LSTMCellStateStub state_before_this_step = current_cell_state_; 
        output_sequence.row(t) = step(input_t, state_before_this_step); 
        // After step, this->current_cell_state_ is updated for the next iteration.
    }
    return output_sequence;
}

void LSTMStub::train_sequence(const Eigen::MatrixXd& input_sequence, 
                              const Eigen::MatrixXd& target_output_sequence, 
                              double learning_rate) {
    if (input_dim_ <= 0 || hidden_dim_ <= 0 || output_dim_ <= 0 ||
        input_sequence.rows() != target_output_sequence.rows() ||
        input_sequence.cols() != input_dim_ || target_output_sequence.cols() != output_dim_ ||
        input_sequence.rows() == 0) {
        // std::cerr << "LSTMStub Train Error: Invalid dimensions or empty sequence." << std::endl;
        last_calculated_loss_ = 1e5; // High loss
        return;
    }
    
    // STUB for BPTT (Backpropagation Through Time)
    // This is highly complex to implement fully. A real implementation would use autograd libraries.
    // For this stub:
    // 1. Perform a forward pass to get outputs and intermediate states (conceptual).
    // 2. Calculate loss (e.g., MSE).
    // 3. Simulate weight updates based on a conceptual gradient.

    Eigen::MatrixXd predicted_outputs = process_sequence(input_sequence); // Resets internal state
    
    Eigen::MatrixXd error_matrix = predicted_outputs - target_output_sequence;
    last_calculated_loss_ = error_matrix.squaredNorm() / static_cast<double>(error_matrix.size()); // MSE

    // Simulate gradient descent update (highly simplified, not real BPTT)
    // Update W_y_ based on the average error of the last hidden states in the sequence
    if (W_y_.size() > 0 && current_cell_state_.hidden_state.size() > 0 && error_matrix.rows() > 0) {
         Eigen::VectorXd avg_output_error = error_matrix.colwise().mean(); // Average error per output dimension
         Eigen::MatrixXd grad_Wy_approx = avg_output_error * current_cell_state_.hidden_state.transpose(); // (OutDim x 1) * (1 x HiddenDim) = OutDim x HiddenDim
         if (grad_Wy_approx.rows() == W_y_.rows() && grad_Wy_approx.cols() == W_y_.cols()) {
            W_y_ -= learning_rate * grad_Wy_approx;
         }
    }
    // Similarly, other weights (W_f, U_f etc.) would be updated. This is a major simplification.
    // For example, conceptually update one set of input weights (W_i)
    if (W_i_.size() > 0 && input_sequence.cols() > 0 && error_matrix.cols() > 0) {
        Eigen::VectorXd avg_input_for_grad = input_sequence.colwise().mean(); // (1 x InputDim)
        // This gradient approximation is not correct for LSTM internal gates but serves as a placeholder
        Eigen::MatrixXd grad_Wi_approx = current_cell_state_.hidden_state * avg_input_for_grad.transpose(); // (HiddenDim x 1) * (1 x InputDim)
        if (grad_Wi_approx.rows() == W_i_.rows() && grad_Wi_approx.cols() == W_i_.cols()) {
             W_i_ -= learning_rate * grad_Wi_approx * 0.1; // Smaller update for internal weights
        }
    }
}

double LSTMStub::get_last_loss() const {
    return last_calculated_loss_;
}
// eane_cpp_modules/learning_module/q_learning_agent_stub.h
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <random> // For random choice in epsilon-greedy

class QLearningAgentStub {
public:
    QLearningAgentStub(int num_states, int num_actions, 
                       double learning_rate = 0.1, double discount_factor = 0.9, 
                       double epsilon = 0.1);

    // Choose action based on current state using epsilon-greedy strategy
    int choose_action(int state);

    // Update Q-value for a state-action pair
    void update(int state, int action, double reward, int next_state);

    // Get Q-value (for inspection or direct use if needed)
    double get_q_value(int state, int action) const;
    int get_num_states() const;
    int get_num_actions() const;

private:
    int num_states_;
    int num_actions_;
    double alpha_; // Learning rate
    double gamma_; // Discount factor
    double epsilon_; // Exploration rate

    Eigen::MatrixXd Q_table_; // Rows: states, Cols: actions

    // Random number generation for epsilon-greedy
    mutable std::mt19937 rng_q_;
    mutable std::uniform_real_distribution<double> uniform_01_q_;
    mutable std::uniform_int_distribution<int> action_dist_q_; // For random action choice
};
// eane_cpp_modules/learning_module/q_learning_agent_stub.cpp
#include "q_learning_agent_stub.h"
#include <algorithm> // std::max_element
#include <iostream>  // For potential debug

QLearningAgentStub::QLearningAgentStub(int num_states, int num_actions, 
                                     double learning_rate, double discount_factor, 
                                     double epsilon)
    : num_states_(num_states), num_actions_(num_actions),
      alpha_(learning_rate), gamma_(discount_factor), epsilon_(epsilon),
      rng_q_(std::random_device{}()), uniform_01_q_(0.0, 1.0) {

    if (num_states_ <= 0 || num_actions_ <= 0) {
        // Handle invalid dimensions. Q_table_ will be 0x0.
        // std::cerr << "QLearningAgentStub Warning: Invalid number of states or actions." << std::endl;
        num_states_ = std::max(0, num_states);
        num_actions_ = std::max(0, num_actions);
    }
    Q_table_ = Eigen::MatrixXd::Zero(num_states_, num_actions_);
    if (num_actions_ > 0) {
        action_dist_q_ = std::uniform_int_distribution<int>(0, num_actions_ - 1);
    }
}

int QLearningAgentStub::choose_action(int state) {
    if (num_actions_ <= 0 || state < 0 || state >= num_states_) {
        // Invalid state or no actions, return a default or error indicator
        return 0; // Or -1 to indicate error
    }

    if (uniform_01_q_(rng_q_) < epsilon_) {
        // Explore: choose a random action
        return action_dist_q_(rng_q_);
    } else {
        // Exploit: choose the action with the highest Q-value for the current state
        // Q_table_.row(state) gives a RowVector. Need to find max index.
        Eigen::Index max_col_index;
        Q_table_.row(state).maxCoeff(&max_col_index); // Gets index of max coefficient in the row
        return static_cast<int>(max_col_index);
    }
}

void QLearningAgentStub::update(int state, int action, double reward, int next_state) {
    if (state < 0 || state >= num_states_ ||
        action < 0 || action >= num_actions_ ||
        next_state < 0 || next_state >= num_states_) {
        // Invalid parameters, do not update
        return;
    }

    // Q(s,a) = Q(s,a) + alpha * (reward + gamma * max_a'(Q(s',a')) - Q(s,a))
    double old_q_value = Q_table_(state, action);
    double max_next_q_value = 0.0;
    if (num_actions_ > 0) { // Ensure there are actions to take max over
        max_next_q_value = Q_table_.row(next_state).maxCoeff();
    }
    
    double new_q_value = old_q_value + alpha_ * (reward + gamma_ * max_next_q_value - old_q_value);
    Q_table_(state, action) = new_q_value;
}

double QLearningAgentStub::get_q_value(int state, int action) const {
    if (state < 0 || state >= num_states_ || action < 0 || action >= num_actions_) {
        return 0.0; // Or throw error, depending on desired behavior for invalid access
    }
    return Q_table_(state, action);
}
int QLearningAgentStub::get_num_states() const { return num_states_;}
int QLearningAgentStub::get_num_actions() const { return num_actions_;}
// eane_cpp_modules/learning_module/knowledge_base_stub.h
#pragma once
#include <string>
#include <map>
#include <any>         // For flexible content storage
#include <vector>
#include <deque>
#include <Eigen/Dense> // For optional vector representation

struct KBEntryStub {
    std::string id;
    std::map<std::string, std::any> content; // Flexible content (e.g., summary, details_ref, metrics)
    std::optional<Eigen::VectorXd> vector_representation; // Optional semantic embedding
    double timestamp_stored;
    // Could add tags, relevance scores, etc.
};

class KnowledgeBaseStub {
public:
    explicit KnowledgeBaseStub(size_t max_size = 1000);

    bool store(const std::string& id, const std::map<std::string, std::any>& content,
               const std::optional<Eigen::VectorXd>& vector_repr = std::nullopt);
    
    std::optional<KBEntryStub> retrieve(const std::string& id) const;
    
    // Conceptual search by vector similarity (very basic stub)
    std::vector<std::pair<std::string, double>> search_by_vector_similarity_stub(
        const Eigen::VectorXd& query_vector, int top_k = 5) const;

    size_t size() const;
    bool empty() const;
    void clear();

private:
    std::map<std::string, KBEntryStub> storage_; // Main storage by ID
    std::deque<std::string> lru_order_;          // For managing max_size (Least Recently Used)
    size_t max_size_;

    void enforce_max_size_();
};
// eane_cpp_modules/learning_module/knowledge_base_stub.cpp
#include "knowledge_base_stub.h"
#include <chrono>    // For timestamps
#include <algorithm> // For std::find, std::remove

KnowledgeBaseStub::KnowledgeBaseStub(size_t max_size) : max_size_(max_size) {}

bool KnowledgeBaseStub::store(const std::string& id, const std::map<std::string, std::any>& content,
                              const std::optional<Eigen::VectorXd>& vector_repr) {
    if (id.empty()) {
        return false; // ID cannot be empty
    }

    KBEntryStub entry;
    entry.id = id;
    entry.content = content;
    entry.vector_representation = vector_repr;
    entry.timestamp_stored = std::chrono::duration<double>(
                                 std::chrono::system_clock::now().time_since_epoch()
                             ).count();

    // If entry exists, update it and move to front of LRU
    auto it = storage_.find(id);
    if (it != storage_.end()) {
        it->second = entry; // Update existing entry
        // Remove from current position in LRU and add to front
        lru_order_.erase(std::remove(lru_order_.begin(), lru_order_.end(), id), lru_order_.end());
    } else {
        storage_[id] = entry;
    }
    
    lru_order_.push_front(id); // New or updated entry is most recently used
    enforce_max_size_();
    return true;
}

std::optional<KBEntryStub> KnowledgeBaseStub::retrieve(const std::string& id) const {
    auto it = storage_.find(id);
    if (it != storage_.end()) {
        // Conceptually, accessing an entry should also update its LRU status,
        // but this const method cannot modify lru_order_. A non-const version would.
        // For a simple stub, we don't implement LRU update on read here.
        return it->second;
    }
    return std::nullopt;
}

std::vector<std::pair<std::string, double>> KnowledgeBaseStub::search_by_vector_similarity_stub(
    const Eigen::VectorXd& query_vector, int top_k) const {
    
    std::vector<std::pair<std::string, double>> results;
    if (query_vector.size() == 0 || storage_.empty()) {
        return results;
    }

    std::vector<std::tuple<double, std::string, const KBEntryStub*>> scored_entries;

    for (const auto& pair_entry : storage_) {
        const KBEntryStub& entry = pair_entry.second;
        if (entry.vector_representation && entry.vector_representation->size() == query_vector.size()) {
            // Cosine similarity: (A dot B) / (||A|| * ||B||)
            double dot_product = query_vector.dot(*(entry.vector_representation));
            double norm_query = query_vector.norm();
            double norm_entry = entry.vector_representation->norm();

            if (norm_query > 1e-9 && norm_entry > 1e-9) {
                double similarity = dot_product / (norm_query * norm_entry);
                scored_entries.emplace_back(-similarity, entry.id, &entry); // Store negative for min-heap behavior with sort
            }
        }
    }

    // Sort by similarity (descending, because we stored negative similarity)
    std::sort(scored_entries.begin(), scored_entries.end(), 
              [](const auto& a, const auto& b) {
                  return std::get<0>(a) < std::get<0>(b); // Sorts by first element (negative similarity)
              });

    int count = 0;
    for (const auto& scored_entry_tuple : scored_entries) {
        if (count >= top_k) break;
        results.emplace_back(std::get<1>(scored_entry_tuple), -std::get<0>(scored_entry_tuple)); // Restore positive similarity
        count++;
    }
    return results;
}

size_t KnowledgeBaseStub::size() const {
    return storage_.size();
}

bool KnowledgeBaseStub::empty() const {
    return storage_.empty();
}

void KnowledgeBaseStub::clear() {
    storage_.clear();
    lru_order_.clear();
}

void KnowledgeBaseStub::enforce_max_size_() {
    while (storage_.size() > max_size_ && !lru_order_.empty()) {
        std::string id_to_remove = lru_order_.back();
        lru_order_.pop_back();
        storage_.erase(id_to_remove);
    }
}
// eane_cpp_modules/learning_module/learning_module.h
#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <map>
#include <any>
#include <deque>
#include <functional>
#include <variant> // For the data in training requests if needed
#include <optional> // For optional return values
#include "../core_interface.h"
#include "lstm_stub.h"
#include "q_learning_agent_stub.h"
#include "knowledge_base_stub.h"

// Forward declare if these are complex types defined elsewhere
// struct MLTrainingRequestContentCpp;
// struct ESSTrainingDataPointCpp;

// Using variant for ML model storage to hold different model types (conceptually)
// In a real system, this would be more structured, possibly with a base class for models.
using MLModelVariantCpp = std::variant<
    std::monostate, // Empty
    Eigen::VectorXd, // e.g., Linear Regression coefficients
    Eigen::MatrixXd, // e.g., K-Means centroids, PCA components
    std::map<std::string, Eigen::MatrixXd> // e.g., ANN weights (layer_name -> weight_matrix)
    // Add more complex model types or custom structs here
>;


class LearningModule {
public:
    LearningModule(CoreInterface* core,
                   int input_dim_lstm_base = 10, int hidden_dim_lstm_base = 20, int output_dim_lstm_base = 5,
                   int num_states_q_base = 10, int num_actions_q_base = 4,
                   size_t kb_max_size = 2000); // Added kb_max_size

    void update_logic();

    // --- Public API for Learning Tasks ---
    void initiate_learning_on_topic(const std::string& topic_query, const std::string& source = "internal_directive_cpp");
    
    // --- Public API for ML/DL Model Training & Prediction (Conceptual/Stubs) ---
    // Return type could be a struct or map holding model_id and performance_metrics.
    // Using std::map for simplicity here.
    using ModelTrainResultCpp = std::map<std::string, std::variant<std::string, double, std::map<std::string, double>>>;

    ModelTrainResultCpp train_supervised_model_conceptual(
        const Eigen::MatrixXd& data_X, const Eigen::VectorXd& data_y,
        const std::string& model_type_str,
        const std::map<std::string, std::any>& params = {}); // `params` for hyperparameters

    ModelTrainResultCpp train_unsupervised_model_conceptual(
        const Eigen::MatrixXd& data_X, const std::string& model_type_str,
        const std::map<std::string, std::any>& params = {});
    
    ModelTrainResultCpp train_ann_conceptual( // Artificial Neural Network
        const Eigen::MatrixXd& data_X, const Eigen::VectorXd& data_y, // data_y for supervised ANN
        const std::string& ann_type_str,
        const std::map<std::string, std::any>& params = {});
    
    ModelTrainResultCpp train_autoencoder_conceptual(
        const Eigen::MatrixXd& data_X, int encoding_dim);

    // Conceptual contrastive learning
    // ModelTrainResultCpp perform_contrastive_learning_conceptual(
    //     const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& data_pairs_similar,
    //     const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& data_pairs_different,
    //     int embedding_dim = 64);

    // Pseudo-labeling (conceptual)
    // std::pair<Eigen::MatrixXd, Eigen::VectorXi> generate_pseudo_labels_conceptual(
    //     const Eigen::MatrixXd& unlabeled_data, const std::string& trained_model_id,
    //     double confidence_threshold = 0.8);

    // Generic prediction (stub, assumes model exists and input_data is appropriate)
    Eigen::VectorXd predict_with_model_stub(const std::string& model_id, const Eigen::MatrixXd& input_data);


    // --- Public API for ESS Integration (Stubs) ---
    // `training_data` would be a vector of structs/maps representing featurized (mutation, scenario, context) -> outcome
    std::string train_ess_vulnerability_predictor_stub(
        const std::vector<std::map<std::string, std::any>>& training_data_sim, // Simplified input
        const std::map<std::string, std::any>& model_config_params = {});
    
    // Predicts <prob_vulnerable, confidence_in_prediction>
    std::pair<double, double> predict_vulnerability_for_ess_stub(
        const std::string& model_id,
        const std::vector<double>& mutation_features,
        const std::vector<double>& scenario_features,
        const std::vector<double>& context_features_gs);
    
    // Featurization functions (can be called by ESS or other modules if LM centralizes this)
    std::vector<double> featurize_mutation_for_ess_model_stub(const std::map<std::string, std::any>& mc_data_sim);
    std::vector<double> featurize_scenario_config_for_ess_model_stub(const std::map<std::string, std::any>& scenario_cfg_data_sim);
    std::vector<double> featurize_system_context_for_ess_model_stub(const GlobalSelfStateCpp& gs_snapshot_sim);

    // --- Accessors for Pybind or internal C++ use ---
    double get_last_lstm_loss() const;
    double get_last_q_reward() const;
    int get_ml_models_count() const;
    int get_learnings_in_kb_count() const;
    // Potentially expose specific model metrics or configurations

private:
    CoreInterface* core_recombinator_;
    LSTMStub lstm_base_stub_; // Renamed from lstm_base_
    QLearningAgentStub q_agent_base_stub_; // Renamed from q_agent_base_
    KnowledgeBaseStub knowledge_base_internal_;

    // ML/DL Model Storage and Tools
    std::map<std::string, MLModelVariantCpp> ml_models_conceptual_;
    // Preprocessing tools are simple functions for now
    std::map<std::string, std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)>> data_preprocessing_tools_conceptual_;

    // ESS Integration State
    std::map<std::string, std::any> featurization_params_ess_config_; // Renamed
    std::string ess_vulnerability_predictor_model_id_;
    // std::string ess_scenario_generator_rl_agent_id_; // RL agent components would go here

    // Internal State (mirroring Python module_state relevant parts)
    std::map<std::string, std::any> current_active_learning_task_details_; // Renamed
    std::map<std::string, std::map<std::string, std::any>> current_active_ml_training_tasks_; // Renamed

    // Metrics to be exposed or logged
    double last_lstm_loss_metric_ = 1.0; // Renamed
    double last_q_reward_metric_ = 0.0;  // Renamed
    double ess_vuln_predictor_accuracy_sim_metric_ = 0.0; // Renamed

    // Internal Helper Methods
    void initialize_internal_preprocessing_tools(); // Renamed
    Eigen::MatrixXd integrate_data_for_lstm_internal(const Eigen::MatrixXd& external_data, const Eigen::MatrixXd& internal_data); // Renamed
    double train_q_learning_cycle_internal(int episodes = 5, int steps_per_episode = 5); // Renamed

    // Event processing and learning cycle dispatchers
    void process_pending_core_events_stub(); // Renamed
    void handle_ml_model_training_request_internal(const EventDataCpp& event_data); // Renamed
    void handle_ess_model_training_request_internal_stub(const EventDataCpp& event_data); // Renamed
    void perform_active_topic_learning_cycle_internal_stub(); // Renamed
    void perform_general_learning_cycle_internal(); // Renamed

    // Random number generation for internal simulations/stubs
    mutable std::mt19937 rng_lm_;
    mutable std::uniform_real_distribution<double> uniform_dist_lm_01_; // Renamed for clarity
    mutable std::normal_distribution<double> normal_dist_lm_std_;   // Renamed for clarity
};
// eane_cpp_modules/learning_module/learning_module.cpp
#include "learning_module.h"
#include <numeric>   // For std::accumulate
#include <algorithm> // For std::transform, std::generate
#include <cmath>     // For std::exp, std::tanh, std::clamp
#include <stdexcept> // For potential errors

LearningModule::LearningModule(CoreInterface* core,
                               int input_dim_lstm_base, int hidden_dim_lstm_base, int output_dim_lstm_base,
                               int num_states_q_base, int num_actions_q_base,
                               size_t kb_max_size)
    : core_recombinator_(core),
      lstm_base_stub_(input_dim_lstm_base, hidden_dim_lstm_base, output_dim_lstm_base),
      q_agent_base_stub_(num_states_q_base, num_actions_q_base),
      knowledge_base_internal_(kb_max_size),
      rng_lm_(std::random_device{}()), 
      uniform_dist_lm_01_(0.0, 1.0), 
      normal_dist_lm_std_(0.0, 1.0) {

    // Initialize ESS featurization parameters (default values)
    featurization_params_ess_config_["mutation_feature_vector_size"] = 20;
    featurization_params_ess_config_["scenario_feature_vector_size"] = 15;
    featurization_params_ess_config_["context_feature_vector_size"] = 15;
    featurization_params_ess_config_["max_categories_one_hot_sim"] = 25; // Conceptual
    // featurization_params_ess_config_["text_embedding_model_for_features_stub_cpp"] = "sbert_MiniLM_L6_v2_sim_cpp_conceptual";

    initialize_internal_preprocessing_tools();
    
    if (core_recombinator_) {
        core_recombinator_->log_message("INFO", "LearningModuleCpp", 
            "LearningModule C++ (V1.1 ESS Integrated - Stub/Conceptual) initialized. LSTM Base Dims: (" +
            std::to_string(input_dim_lstm_base) + "," + std::to_string(hidden_dim_lstm_base) + "," + std::to_string(output_dim_lstm_base) + 
            "). Q-Agent Base: (S:" + std::to_string(num_states_q_base) + ", A:" + std::to_string(num_actions_q_base) + ")");
    }
}

void LearningModule::initialize_internal_preprocessing_tools() {
    // Simple standard scaler (centers to mean 0, scales to unit variance)
    data_preprocessing_tools_conceptual_["standard_scaler_sim_cpp"] =
        [](const Eigen::MatrixXd& data_in) -> Eigen::MatrixXd {
        if (data_in.rows() == 0 || data_in.cols() == 0) return data_in;
        Eigen::MatrixXd data = data_in; // Modifiable copy
        Eigen::VectorXd mean = data.colwise().mean();
        data.rowwise() -= mean.transpose(); // Center data
        
        Eigen::VectorXd std_dev(data.cols());
        for (int j = 0; j < data.cols(); ++j) {
            double variance = data.col(j).array().square().sum() / std::max(1.0, static_cast<double>(data.rows() -1)); // Sample variance
            std_dev(j) = std::sqrt(variance);
            if (std_dev(j) < 1e-9) { // Avoid division by zero or very small std dev
                std_dev(j) = 1.0;
            }
        }
        data.array().rowwise() /= std_dev.transpose().array(); // Scale
        return data;
    };

    // Simple Min-Max scaler (scales to [0, 1] range)
    data_preprocessing_tools_conceptual_["min_max_scaler_sim_cpp"] =
        [](const Eigen::MatrixXd& data_in) -> Eigen::MatrixXd {
        if (data_in.rows() == 0 || data_in.cols() == 0) return data_in;
        Eigen::MatrixXd data = data_in; // Modifiable copy
        Eigen::VectorXd min_vals = data.colwise().minCoeff();
        Eigen::VectorXd max_vals = data.colwise().maxCoeff();
        Eigen::VectorXd range_vals = max_vals - min_vals;

        for (int j = 0; j < data.cols(); ++j) {
            if (range_vals(j) < 1e-9) { // If all values in column are same
                range_vals(j) = 1.0; // Avoid division by zero, results in 0s if min=max
            }
        }
        data.rowwise() -= min_vals.transpose();
        data.array().rowwise() /= range_vals.transpose().array();
        return data;
    };
    // Add more conceptual tools like TF-IDF vectorizer stub if needed
}


void LearningModule::update_logic() {
    if (!core_recombinator_) return;

    process_pending_core_events_stub(); // Process incoming requests or data

    if (!current_active_learning_task_details_.empty()) {
        perform_active_topic_learning_cycle_internal_stub();
    } else {
        perform_general_learning_cycle_internal();
    }

    // Conceptual: Periodically check for self-improvement needs
    // if (core_recombinator_->get_current_cycle_num() % 150 == 0) {
    //     if (last_lstm_loss_metric_ > 0.35 || ess_vuln_predictor_accuracy_sim_metric_ < 0.60) {
    //          EventDataCpp self_improve_req("module_self_improvement_request_cpp");
    //          self_improve_req.source_module = "LearningModuleCpp";
    //          self_improve_req.content["area_of_concern_str"] = std::string("suboptimal_learning_model_performance_cpp");
    //          // ... add more metrics to content ...
    //          core_recombinator_->event_queue_put(self_improve_req);
    //     }
    // }
}

void LearningModule::process_pending_core_events_stub() {
    // This is a STUB. In a real system, this would interact with CoreInterface's event queue.
    // Example:
    // EventDataCpp event_data;
    // if (core_recombinator_->event_queue_get_specific_list_cpp(
    //         {"new_learning_task_for_lm_cpp", "lm_train_ml_model_request_cpp", "lm_train_model_for_ess_request_cpp"},
    //         event_data, 0.001 /* timeout */)) {
    //
    //     if (event_data.type == "new_learning_task_for_lm_cpp") {
    //         // Extract topic and source, then call initiate_learning_on_topic
    //     } else if (event_data.type == "lm_train_ml_model_request_cpp") {
    //         handle_ml_model_training_request_internal(event_data);
    //     } else if (event_data.type == "lm_train_model_for_ess_request_cpp") {
    //         handle_ess_model_training_request_internal_stub(event_data);
    //     }
    // }
}

void LearningModule::handle_ml_model_training_request_internal(const EventDataCpp& event_data) {
    // STUB: Parse event_data.content to get MLTrainingRequestContentCpp details.
    // This involves careful casting from std::variant/std::any.
    // Then, call the appropriate train_..._model_conceptual method.
    // Finally, send a "lm_ml_model_training_completed_cpp" event back.
    if (core_recombinator_) core_recombinator_->log_message("DEBUG", "LearningModuleCpp", "STUB: Handling ML model training request for type: " + event_data.type);
    
    // Conceptual extraction (replace with robust variant/any casting)
    std::string model_type_req = "unknown_model_cpp";
    std::string task_type_ml = "unknown_task_cpp";
    std::string requesting_module = "unknown_requester_cpp";

    if(event_data.content.count("model_type_request_str")){ // Assume string for simplicity
        try { model_type_req = std::get<std::string>(event_data.content.at("model_type_request_str")); } catch(const std::bad_variant_access&){}
    }
    // ... similar for task_type, data IDs (if used), hyperparameters ...

    // Simulate getting data (this would involve DKPM in Python version)
    Eigen::MatrixXd X_sim = Eigen::MatrixXd::Random(50, 5); // 50 samples, 5 features
    Eigen::VectorXd y_sim = Eigen::VectorXd::Random(50);    // For supervised

    ModelTrainResultCpp train_result;
    if (task_type_ml.find("supervised") != std::string::npos) {
        train_result = train_supervised_model_conceptual(X_sim, y_sim, model_type_req);
    } else if (task_type_ml.find("unsupervised") != std::string::npos) {
        train_result = train_unsupervised_model_conceptual(X_sim, model_type_req);
    } // ... more cases for ANN, Autoencoder ...

    // Send completion event (conceptual)
    // EventDataCpp completion_event("lm_ml_model_training_completed_cpp");
    // completion_event.target_module = requesting_module;
    // ... populate with train_result ...
    // core_recombinator_->event_queue_put(completion_event);
}

// ... (Implementation of LearningModule methods will continue in the next part) ...
// eane_cpp_modules/learning_module/learning_module.cpp
// (Continuation from Part 5)
#include "learning_module.h" // Already included
#include <map> // For ModelTrainResultCpp

// --- Public API for Learning Tasks ---
void LearningModule::initiate_learning_on_topic(const std::string& topic_query, const std::string& source) {
    if (!core_recombinator_) return;
    
    current_active_learning_task_details_["topic_str"] = topic_query; // Use _str for simple map
    current_active_learning_task_details_["source_str"] = source;
    current_active_learning_task_details_["status_str"] = std::string("pending_cpp_topic_learn");

    core_recombinator_->log_message("INFO", "LearningModuleCpp", 
        "Initiating learning on topic (C++): '" + topic_query.substr(0, 50) + "' from source: " + source);

    EventDataCpp task_event("new_learning_task_for_lm_py_counterpart"); // To inform Python side if needed
    task_event.content["topic_str"] = topic_query;
    task_event.content["source_str"] = source;
    task_event.content["status_str"] = std::string("pending_cpp_topic_learn");
    task_event.priority_label = "medium";
    core_recombinator_->event_queue_put(task_event);

    // Conceptual: If source is "external_web_cpp", trigger ANA (AdvancedNetworkAnalyzer)
    // This would involve sending an event to ANA.
    // if (source.find("external") != std::string::npos || source.find("web") != std::string::npos) {
    //     EventDataCpp ana_request("ana_data_fetch_request_py_counterpart");
    //     ana_request.content["topic_str"] = topic_query;
    //     ana_request.content["requesting_module_str"] = std::string("LearningModuleCpp");
    //     core_recombinator_->event_queue_put(ana_request);
    //     current_active_learning_task_details_["status_str"] = std::string("data_fetching_requested_to_ana_cpp");
    // }
}

// --- Internal Learning Cycles ---
void LearningModule::perform_active_topic_learning_cycle_internal_stub() {
    if (!core_recombinator_ || current_active_learning_task_details_.empty()) return;

    std::string topic = "unknown_topic_cpp";
    std::string status = "unknown_status_cpp";
    if(current_active_learning_task_details_.count("topic_str")){
        try{ topic = std::any_cast<std::string>(current_active_learning_task_details_["topic_str"]); } catch(const std::bad_any_cast&){}
    }
    if(current_active_learning_task_details_.count("status_str")){
        try{ status = std::any_cast<std::string>(current_active_learning_task_details_["status_str"]); } catch(const std::bad_any_cast&){}
    }

    core_recombinator_->log_message("DEBUG", "LearningModuleCpp",
        "Performing active topic learning cycle (STUB) for: '" + topic.substr(0,30) + "', Status: " + status);

    // STUB: Simulate processing based on status
    // if (status == "data_ready_for_processing_cpp") { // Assume ANA or other module sets this
        // Simulate processing topic data
        // For now, just use the general learning cycle as a placeholder
        perform_general_learning_cycle_internal();

        // Store conceptual learning in KB
        std::string kb_id = "cpp_topic_learned_" + topic.substr(0,20) + "_" + std::to_string(core_recombinator_->get_current_timestamp());
        std::map<std::string, std::any> kb_content;
        kb_content["summary_str"] = std::string("Conceptual knowledge assimilated (C++) about: ") + topic;
        kb_content["details_ref_str"] = std::string("Simulated processing of topic data for ") + topic;
        kb_content["confidence_sim_val"] = uniform_dist_lm_01_(rng_lm_) * 0.3 + 0.5; // Simulated confidence
        knowledge_base_internal_.store(kb_id, kb_content);

        // Event for major learning achievement if confidence is high (conceptual)
        // if (std::any_cast<double>(kb_content["confidence_sim_val"]) > 0.75) {
        //    EventDataCpp major_learn_event("major_learning_achieved_cpp");
        //    major_learn_event.content["summary_str"] = std::string("Achieved major learning on topic: ") + topic;
        //    major_learn_event.content["knowledge_id_in_kb_str"] = kb_id;
        //    core_recombinator_->event_queue_put(major_learn_event);
        // }
        current_active_learning_task_details_.clear(); // Clear task after processing
    // } else if (status == "pending_cpp_topic_learn" || status == "data_fetching_requested_to_ana_cpp") {
        // Waiting for data or next step
    // } else {
        // core_recombinator_->log_message("WARNING", "LearningModuleCpp", "Active topic learning task in unknown status: " + status);
        // current_active_learning_task_details_.clear(); // Clear unknown task
    // }
}

void LearningModule::perform_general_learning_cycle_internal() {
    if (!core_recombinator_) return;
    GlobalSelfStateCpp& gs = core_recombinator_->get_global_state();

    // Ensure LSTM base stub has valid dimensions (input_dim_lstm_base from Python)
    int lstm_input_dim = lstm_base_stub_.get_input_dim(); // Need getters in LSTMStub
    int lstm_output_dim = lstm_base_stub_.get_output_dim();
    if (lstm_input_dim <= 0 || lstm_output_dim <= 0) {
        core_recombinator_->log_message("DEBUG", "LearningModuleCpp", "LSTM base stub not configured for general learning cycle.");
        return;
    }

    Eigen::VectorXd current_global_metrics_vec(lstm_input_dim); // Assuming this matches obs_features size
    // Populate current_global_metrics_vec similar to SubconsciousMind's observation vector
    if (lstm_input_dim == 10) { // Match the 10 features from Python
        double mean_needs = 0.5;
        if (gs.needs_vector.size() >= 3) mean_needs = gs.needs_vector.mean();
        else if (gs.needs_vector.size() > 0) mean_needs = gs.needs_vector.mean();
        
        current_global_metrics_vec << gs.valencia, gs.arousal, gs.motivacion, gs.dolor, 
                               gs.coherence_score, gs.system_entropy, gs.self_esteem,
                               mean_needs, gs.phi_consciousness, gs.resilience_stability_metric;
    } else { // Fallback if lstm_input_dim is not 10
        for(int i=0; i < lstm_input_dim; ++i) current_global_metrics_vec(i) = uniform_dist_lm_01_(rng_lm_); // Random data
    }


    int sequence_length = 5; // Short sequence for stub
    double noise_factor = 0.02;

    Eigen::MatrixXd external_data_sim_seq(sequence_length, lstm_input_dim);
    Eigen::MatrixXd internal_data_sim_seq(sequence_length, lstm_input_dim);

    for (int r = 0; r < sequence_length; ++r) {
        for (int c = 0; c < lstm_input_dim; ++c) {
            external_data_sim_seq(r,c) = current_global_metrics_vec(c) + normal_dist_lm_std_(rng_lm_) * noise_factor;
            internal_data_sim_seq(r,c) = current_global_metrics_vec(c) + normal_dist_lm_std_(rng_lm_) * (noise_factor / 1.5);
        }
    }

    Eigen::MatrixXd target_data_sim_seq(sequence_length, lstm_output_dim);
    if (lstm_output_dim >= 2) {
        for (int i = 0; i < sequence_length; ++i) {
            target_data_sim_seq(i, 0) = gs.coherence_score + normal_dist_lm_std_(rng_lm_) * 0.005;
            target_data_sim_seq(i, 1) = gs.system_entropy + normal_dist_lm_std_(rng_lm_) * 0.005;
            for (int j = 2; j < lstm_output_dim; ++j) target_data_sim_seq(i,j) = uniform_dist_lm_01_(rng_lm_) * 0.05;
        }
    } else if (lstm_output_dim == 1) {
         for (int i = 0; i < sequence_length; ++i) target_data_sim_seq(i,0) = gs.coherence_score + normal_dist_lm_std_(rng_lm_) * 0.005;
    } else {
        target_data_sim_seq.resize(sequence_length, 0); // Empty target if no output dim
    }

    if (target_data_sim_seq.cols() == lstm_output_dim && external_data_sim_seq.cols() == lstm_input_dim) {
        Eigen::MatrixXd combined_data_for_lstm = integrate_data_for_lstm_internal(external_data_sim_seq, internal_data_sim_seq);
        
        // Simulate LSTM training (2 epochs for stub)
        for (int epoch = 0; epoch < 2; ++epoch) {
            lstm_base_stub_.train_sequence(combined_data_for_lstm, target_data_sim_seq, 0.003 /*learning_rate*/);
        }
        last_lstm_loss_metric_ = lstm_base_stub_.get_last_loss();

        // Simulate Q-learning cycle
        last_q_reward_metric_ = train_q_learning_cycle_internal(5 /*episodes*/);

        // Store learning in KB
        std::string kb_id = "lm_cpp_gen_cycle_" + std::to_string(core_recombinator_->get_current_cycle_num());
        std::map<std::string, std::any> kb_content;
        kb_content["type_str"] = std::string("general_learning_cycle_cpp_internal");
        kb_content["lstm_loss_base_val"] = last_lstm_loss_metric_;
        kb_content["q_reward_base_val"] = last_q_reward_metric_;
        Eigen::VectorXd kb_embedding_vec(4); // Simple embedding
        kb_embedding_vec << last_lstm_loss_metric_, last_q_reward_metric_, external_data_sim_seq.mean(), target_data_sim_seq.mean();
        knowledge_base_internal_.store(kb_id, kb_content, kb_embedding_vec);

        // EventDataCpp cycle_complete_event("minor_learning_cycle_completed_cpp_internal");
        // ... (populate content) ...
        // core_recombinator_->event_queue_put(cycle_complete_event);
    }
}

Eigen::MatrixXd LearningModule::integrate_data_for_lstm_internal(const Eigen::MatrixXd& external_data, const Eigen::MatrixXd& internal_data) {
    // Normalize and combine data (same as Python version's _integrate_data_lm)
    Eigen::MatrixXd ext_norm = (external_data.array().max(-10.0).min(10.0)) / 10.0;
    Eigen::MatrixXd int_norm = (internal_data.array().max(-10.0).min(10.0)) / 10.0;
    return 0.7 * ext_norm + 0.3 * int_norm;
}

double LearningModule::train_q_learning_cycle_internal(int episodes, int steps_per_episode) {
    if (q_agent_base_stub_.get_num_states() == 0 || !core_recombinator_) return 0.0;

    GlobalSelfStateCpp& gs = core_recombinator_->get_global_state();
    double total_reward_accum = 0.0;
    int num_total_steps = 0;

    std::uniform_int_distribution<int> state_dist(0, q_agent_base_stub_.get_num_states() - 1);

    for (int ep = 0; ep < episodes; ++ep) {
        int current_state_q = state_dist(rng_lm_);
        for (int st = 0; st < steps_per_episode; ++st) {
            int action_q = q_agent_base_stub_.choose_action(current_state_q);
            // Simplified reward based on global state
            double reward_q = gs.valencia * 0.3 + gs.coherence_score * 0.2 - gs.system_entropy * 0.1 + normal_dist_lm_std_(rng_lm_) * 0.05;
            int next_state_q = state_dist(rng_lm_);

            q_agent_base_stub_.update(current_state_q, action_q, reward_q, next_state_q);
            total_reward_accum += reward_q;
            current_state_q = next_state_q;
            num_total_steps++;
        }
    }
    return (num_total_steps > 0) ? total_reward_accum / static_cast<double>(num_total_steps) : 0.0;
}


// --- Public API for ML/DL Model Training & Prediction (Conceptual/Stubs) ---
// (Implementations for train_supervised_model_conceptual, train_unsupervised_model_conceptual, etc.
//  were started in Part 3's .cpp file and would continue here.
//  For brevity, I will provide one more example and then stubs for the rest in this part.)

LearningModule::ModelTrainResultCpp LearningModule::train_supervised_model_conceptual(
    const Eigen::MatrixXd& data_X, const Eigen::VectorXd& data_y,
    const std::string& model_type_str,
    const std::map<std::string, std::any>& params) {
    
    if (!core_recombinator_) return {{"error_str", std::string("Core not available")}};
    std::string task_id = "sup_train_cpp_" + model_type_str + "_" + std::to_string(core_recombinator_->get_current_timestamp());
    // current_active_ml_training_tasks_[task_id] = {{"type_str", ...}, {"status_str", ...}}; // Update task status

    core_recombinator_->log_message("INFO", "LearningModuleCpp", "Training supervised model (C++ Stub): " + model_type_str);

    Eigen::MatrixXd X_processed = data_X;
    if (data_preprocessing_tools_conceptual_.count("standard_scaler_sim_cpp")) {
        X_processed = data_preprocessing_tools_conceptual_["standard_scaler_sim_cpp"](data_X);
    }

    std::string model_id = model_type_str + "_cpp_model_" + std::to_string(core_recombinator_->get_current_timestamp());
    ModelTrainResultCpp result;
    std::map<std::string, double> metrics_map;

    if (model_type_str == "linear_regression_cpp_sim") {
        if (X_processed.rows() < X_processed.cols() || X_processed.rows() != data_y.size()) {
            metrics_map["error_val"] = 1.0; // Error flag
            metrics_map["r2_score_sim_val"] = 0.0;
            result["model_id"] = std::string("failed_") + model_id;
        } else {
            Eigen::VectorXd coeffs = X_processed.colPivHouseholderQr().solve(data_y);
            Eigen::VectorXd y_pred = X_processed * coeffs;
            double ss_res = (data_y - y_pred).squaredNorm();
            double ss_tot = (data_y.array() - data_y.mean()).square().sum();
            metrics_map["r2_score_sim_val"] = (ss_tot > 1e-9) ? (1.0 - ss_res / ss_tot) : 0.0;
            ml_models_conceptual_[model_id] = coeffs; // Store coefficients (Eigen::VectorXd)
            result["model_id"] = model_id;
        }
    } else if (model_type_str == "svm_classification_cpp_sim") {
        metrics_map["accuracy_sim_val"] = uniform_dist_lm_01_(rng_lm_) * 0.3 + 0.65; // Simulated accuracy
        ml_models_conceptual_[model_id] = std::map<std::string, double>{{"svm_complexity_sim_val", uniform_dist_lm_01_(rng_lm_) }}; // Placeholder model
        result["model_id"] = model_id;
    } else {
        metrics_map["error_val"] = 1.0;
        result["model_id"] = std::string("unsupported_") + model_id;
        core_recombinator_->log_message("WARNING", "LearningModuleCpp", "Unsupported supervised model type: " + model_type_str);
    }
    result["metrics_simulated_map"] = metrics_map; // Store metrics map
    return result;
}

// ... (Stubs for train_unsupervised, train_ann, train_autoencoder would follow a similar pattern)
// ... (Implementations for ESS methods will be very conceptual stubs)

Eigen::VectorXd LearningModule::predict_with_model_stub(const std::string& model_id, const Eigen::MatrixXd& input_data) {
    if (ml_models_conceptual_.find(model_id) == ml_models_conceptual_.end()) {
        if (core_recombinator_) core_recombinator_->log_message("WARNING", "LearningModuleCpp", "Predict: Model ID '" + model_id + "' not found.");
        return Eigen::VectorXd(input_data.rows()); // Return vector of zeros of appropriate size
    }
    const auto& model_variant = ml_models_conceptual_.at(model_id);

    // STUB: Very basic prediction logic based on stored model type (conceptual)
    if (std::holds_alternative<Eigen::VectorXd>(model_variant)) { // Assumed Linear Regression Coeffs
        const Eigen::VectorXd& coeffs = std::get<Eigen::VectorXd>(model_variant);
        if (input_data.cols() == coeffs.size()) {
            return input_data * coeffs;
        } else if (input_data.cols() == coeffs.size() -1) { // Maybe intercept term was included
             Eigen::MatrixXd X_with_intercept(input_data.rows(), input_data.cols() + 1);
             X_with_intercept << input_data, Eigen::MatrixXd::Ones(input_data.rows(), 1);
             if (coeffs.size() == X_with_intercept.cols()) return X_with_intercept * coeffs;
        }
    } else if (std::holds_alternative<std::map<std::string, Eigen::MatrixXd>>(model_variant)) { // ANN weights conceptual
        // Simulate ANN forward pass (very rough)
        // For simplicity, just return random values based on expected output dim
        // This needs the model to store output dimension info.
        // const auto& ann_weights = std::get<std::map<std::string, Eigen::MatrixXd>>(model_variant);
        // if (ann_weights.count("W2")) return Eigen::VectorXd::Random(input_data.rows()) * (ann_weights.at("W2").cols());
    } else if (std::holds_alternative<std::map<std::string, double>>(model_variant)) { // SVM placeholder or K-Means (if it stored metrics)
        // Return random classification/clustering
    }
    
    // Default fallback: random predictions
    int num_outputs_guess = 1; // Guess one output if model type is unknown
    if (model_id.find("classification") != std::string::npos) num_outputs_guess = 2; // Binary classification
    return Eigen::VectorXd::Random(input_data.rows() * num_outputs_guess).reshaped(input_data.rows(), num_outputs_guess);
}


// --- Accessors ---
double LearningModule::get_last_lstm_loss() const { return last_lstm_loss_metric_; }
double LearningModule::get_last_q_reward() const { return last_q_reward_metric_; }
int LearningModule::get_ml_models_count() const { return static_cast<int>(ml_models_conceptual_.size()); }
int LearningModule::get_learnings_in_kb_count() const { return static_cast<int>(knowledge_base_internal_.size()); }
// eane_cpp_modules/learning_module/learning_module.cpp
// (Continuation from Part 6)
#include "learning_module.h" // Already included

// --- Stubs for other ML Training Methods ---
LearningModule::ModelTrainResultCpp LearningModule::train_unsupervised_model_conceptual(
    const Eigen::MatrixXd& data_X, const std::string& model_type_str,
    const std::map<std::string, std::any>& params) {
    
    if (!core_recombinator_) return {{"error_str", std::string("Core not available")}};
    std::string task_id = "unsup_train_cpp_" + model_type_str + "_" + std::to_string(core_recombinator_->get_current_timestamp());
    // current_active_ml_training_tasks_[task_id] = { ... }; // Update task status

    core_recombinator_->log_message("INFO", "LearningModuleCpp", "Training unsupervised model (C++ Stub): " + model_type_str);
    
    std::string model_id = model_type_str + "_cpp_model_" + std::to_string(core_recombinator_->get_current_timestamp());
    ModelTrainResultCpp result;
    std::map<std::string, double> metrics_map;

    if (model_type_str == "kmeans_clustering_cpp_sim") {
        int n_clusters = 3; // Default
        if(params.count("n_clusters_val")){ try{ n_clusters = std::any_cast<int>(params.at("n_clusters_val")); } catch(const std::bad_any_cast&){} }
        
        if (data_X.rows() < n_clusters || data_X.cols() == 0) {
            metrics_map["error_val"] = 1.0;
            result["model_id"] = std::string("failed_") + model_id;
        } else {
            metrics_map["n_clusters_found_sim_val"] = static_cast<double>(n_clusters);
            metrics_map["silhouette_score_sim_val"] = uniform_dist_lm_01_(rng_lm_) * 0.4 + 0.3; // Simulated score
            ml_models_conceptual_[model_id] = Eigen::MatrixXd::Random(n_clusters, data_X.cols()); // Store centroids
            result["model_id"] = model_id;
        }
    } else if (model_type_str == "pca_cpp_sim") {
        int n_components = 2; // Default
        if(params.count("n_components_val")){ try{ n_components = std::any_cast<int>(params.at("n_components_val")); } catch(const std::bad_any_cast&){} }

        if (data_X.cols() == 0 || n_components <= 0 || n_components > data_X.cols()) {
             metrics_map["error_val"] = 1.0;
             result["model_id"] = std::string("failed_") + model_id;
        } else {
            metrics_map["explained_variance_ratio_sum_sim_val"] = uniform_dist_lm_01_(rng_lm_) * 0.5 + 0.4;
            ml_models_conceptual_[model_id] = Eigen::MatrixXd::Random(data_X.cols(), n_components); // Store components
            result["model_id"] = model_id;
        }
    }
    else {
        metrics_map["error_val"] = 1.0;
        result["model_id"] = std::string("unsupported_") + model_id;
    }
    result["metrics_simulated_map"] = metrics_map;
    return result;
}

LearningModule::ModelTrainResultCpp LearningModule::train_ann_conceptual(
    const Eigen::MatrixXd& data_X, const Eigen::VectorXd& data_y,
    const std::string& ann_type_str,
    const std::map<std::string, std::any>& params) {

    if (!core_recombinator_) return {{"error_str", std::string("Core not available")}};
    core_recombinator_->log_message("INFO", "LearningModuleCpp", "Training ANN (C++ Stub): " + ann_type_str);
    
    std::string model_id = ann_type_str + "_cpp_model_" + std::to_string(core_recombinator_->get_current_timestamp());
    ModelTrainResultCpp result;
    std::map<std::string, double> metrics_map;

    if (data_X.rows() != data_y.size() || data_X.rows() == 0) {
        metrics_map["error_val"] = 1.0;
        result["model_id"] = std::string("failed_") + model_id;
    } else {
        bool is_classification = ann_type_str.find("classification") != std::string::npos;
        if (is_classification) {
            metrics_map["accuracy_sim_val"] = uniform_dist_lm_01_(rng_lm_) * 0.3 + 0.65;
        } else { // Regression
            metrics_map["mse_sim_val"] = uniform_dist_lm_01_(rng_lm_) * 0.15 + 0.05;
        }
        // Store conceptual weights
        std::map<std::string, Eigen::MatrixXd> ann_weights;
        ann_weights["W1_sim"] = Eigen::MatrixXd::Random(data_X.cols(), 32); // Input to Hidden1
        ann_weights["W2_sim"] = Eigen::MatrixXd::Random(32, 1); // Hidden1 to Output (assuming 1 output for simplicity)
        ml_models_conceptual_[model_id] = ann_weights;
        result["model_id"] = model_id;
    }
    result["metrics_simulated_map"] = metrics_map;
    return result;
}

LearningModule::ModelTrainResultCpp LearningModule::train_autoencoder_conceptual(
    const Eigen::MatrixXd& data_X, int encoding_dim) {

    if (!core_recombinator_) return {{"error_str", std::string("Core not available")}};
    core_recombinator_->log_message("INFO", "LearningModuleCpp", "Training Autoencoder (C++ Stub), Encoding Dim: " + std::to_string(encoding_dim));

    std::string model_id = "autoencoder_cpp_model_" + std::to_string(core_recombinator_->get_current_timestamp());
    ModelTrainResultCpp result;
    std::map<std::string, double> metrics_map;

    if (data_X.cols() == 0 || encoding_dim <= 0 || encoding_dim >= data_X.cols()) {
        metrics_map["error_val"] = 1.0;
        result["model_id"] = std::string("failed_") + model_id;
    } else {
        metrics_map["reconstruction_error_sim_val"] = uniform_dist_lm_01_(rng_lm_) * 0.1 + 0.01;
        // Store conceptual encoder/decoder structure (e.g., dimensions)
        std::map<std::string, int> ae_config;
        ae_config["original_dim_val"] = static_cast<int>(data_X.cols());
        ae_config["encoding_dim_val"] = encoding_dim;
        ml_models_conceptual_[model_id] = ae_config; // Store config as the "model"
        result["model_id"] = model_id;
    }
    result["metrics_simulated_map"] = metrics_map;
    return result;
}

// --- Stubs for ESS Integration Methods ---
std::vector<double> LearningModule::featurize_mutation_for_ess_model_stub(const std::map<std::string, std::any>& mc_data_sim) {
    int vec_size = 20; // Default
    if(featurization_params_ess_config_.count("mutation_feature_vector_size")){
        try{ vec_size = std::any_cast<int>(featurization_params_ess_config_.at("mutation_feature_vector_size")); } catch(const std::bad_any_cast&){}
    }
    std::vector<double> features(vec_size);
    std::generate(features.begin(), features.end(), [&](){ return uniform_dist_lm_01_(rng_lm_); });
    return features;
}

std::vector<double> LearningModule::featurize_scenario_config_for_ess_model_stub(const std::map<std::string, std::any>& scenario_cfg_data_sim) {
    int vec_size = 15; // Default
    if(featurization_params_ess_config_.count("scenario_feature_vector_size")){
         try{ vec_size = std::any_cast<int>(featurization_params_ess_config_.at("scenario_feature_vector_size")); } catch(const std::bad_any_cast&){}
    }
    std::vector<double> features(vec_size);
    std::generate(features.begin(), features.end(), [&](){ return uniform_dist_lm_01_(rng_lm_); });
    return features;
}

std::vector<double> LearningModule::featurize_system_context_for_ess_model_stub(const GlobalSelfStateCpp& gs_snapshot_sim) {
    int vec_size = 15; // Default
     if(featurization_params_ess_config_.count("context_feature_vector_size")){
         try{ vec_size = std::any_cast<int>(featurization_params_ess_config_.at("context_feature_vector_size")); } catch(const std::bad_any_cast&){}
    }
    std::vector<double> features;
    features.reserve(vec_size);
    features.push_back(gs_snapshot_sim.valencia);
    features.push_back(gs_snapshot_sim.arousal);
    features.push_back(gs_snapshot_sim.system_entropy);
    features.push_back(gs_snapshot_sim.coherence_score);
    // ... add more key GS metrics ...
    while(features.size() < static_cast<size_t>(vec_size)) {
        features.push_back(uniform_dist_lm_01_(rng_lm_) * 0.3); // Pad with small random values
    }
    return std::vector<double>(features.begin(), features.begin() + std::min(features.size(), static_cast<size_t>(vec_size))); // Ensure correct size
}


std::string LearningModule::train_ess_vulnerability_predictor_stub(
    const std::vector<std::map<std::string, std::any>>& training_data_sim,
    const std::map<std::string, std::any>& model_config_params) {
    
    if (!core_recombinator_) return "error_no_core_cpp";
    if (training_data_sim.empty()) {
        core_recombinator_->log_message("WARNING", "LearningModuleCpp", "ESS Vuln Predictor (Stub): No training data provided.");
        return ess_vulnerability_predictor_model_id_; // Return existing if any
    }
    core_recombinator_->log_message("INFO", "LearningModuleCpp", 
        "Training ESS Vulnerability Predictor (C++ Stub) with " + std::to_string(training_data_sim.size()) + " data points.");

    // Simulate featurization and training
    // In a real system, this would call featurize_... methods and then train_supervised_model_conceptual
    ess_vuln_predictor_accuracy_sim_metric_ = uniform_dist_lm_01_(rng_lm_) * 0.25 + 0.6; // Simulate new accuracy
    ess_vulnerability_predictor_model_id_ = "ess_vuln_pred_cpp_model_" + std::to_string(core_recombinator_->get_current_timestamp());
    ml_models_conceptual_[ess_vulnerability_predictor_model_id_] = std::map<std::string, double>{
        {"type_str_val", ess_vuln_predictor_accuracy_sim_metric_} // Store accuracy as part of the "model"
    };
    
    core_recombinator_->log_message("INFO", "LearningModuleCpp", 
        "ESS Vulnerability Predictor (Stub) updated: " + ess_vulnerability_predictor_model_id_ + 
        ". Simulated Accuracy: " + std::to_string(ess_vuln_predictor_accuracy_sim_metric_));
    return ess_vulnerability_predictor_model_id_;
}

std::pair<double, double> LearningModule::predict_vulnerability_for_ess_stub(
    const std::string& model_id,
    const std::vector<double>& mutation_features,
    const std::vector<double>& scenario_features,
    const std::vector<double>& context_features_gs) {
    
    if (model_id.empty() || model_id != ess_vulnerability_predictor_model_id_ ||
        ml_models_conceptual_.find(model_id) == ml_models_conceptual_.end()) {
        if(core_recombinator_) core_recombinator_->log_message("WARNING", "LearningModuleCpp", "ESS Predict Vuln (Stub): Invalid or unknown model ID: " + model_id);
        return {uniform_dist_lm_01_(rng_lm_) * 0.4, 0.2}; // Low confidence random prediction
    }
    // Simulate prediction
    double prob_vulnerable_sim = uniform_dist_lm_01_(rng_lm_); // Random probability
    double confidence_sim = ess_vuln_predictor_accuracy_sim_metric_ * (uniform_dist_lm_01_(rng_lm_) * 0.3 + 0.7); // Confidence related to model accuracy
    return {prob_vulnerable_sim, confidence_sim};
}

void LearningModule::handle_ess_model_training_request_internal_stub(const EventDataCpp& event_data) {
    // STUB: Parse event, call appropriate ESS training method
    if (core_recombinator_) core_recombinator_->log_message("DEBUG", "LearningModuleCpp", "STUB: Handling ESS model training request for type: " + event_data.type);

    // Conceptual:
    // std::string model_purpose = ... get from event_data.content ...
    // std::vector<std::map<std::string, std::any>> training_data = ...
    // if (model_purpose == "vulnerability_predictor_cpp") {
    //    train_ess_vulnerability_predictor_stub(training_data, {});
    // } else if (model_purpose == "scenario_generation_rl_agent_cpp") {
    //    // ... train RL agent stub ...
    // }
}
// eane_cpp_modules/self_evolution_module/sem_types.h
#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <map>
#include <optional> // C++17

// --- Individual Representation ---
struct IndividualCpp {
    Eigen::VectorXd parameters; // Abstract genome
    double fitness = -1e38; // Using a very small number for uninitialized
    double novelty_score = 0.0;
    std::map<std::string, double> secondary_objectives_scores; // For Multi-Objective EA
    int age_generations = 0;
    std::optional<std::pair<std::string, std::string>> parent_ids_sim; // Conceptual IDs
    std::vector<std::string> mutation_ids_applied_sim; // Conceptual IDs
    std::string id; // Unique ID for the individual

    explicit IndividualCpp(int genome_dim) : parameters(Eigen::VectorXd::Zero(genome_dim)) {
        static long long id_counter = 0;
        id = "sem_ind_cpp_" + std::to_string(id_counter++);
    }
    // Constructor to initialize with parameters
    IndividualCpp(const Eigen::VectorXd& params) : parameters(params) {
        static long long id_counter = 0;
        id = "sem_ind_cpp_" + std::to_string(id_counter++);
    }
};

// --- Fitness Landscape Configuration ---
struct ObjectiveDefinitionCpp {
    std::string metric_path; // e.g., "gs.coherence_score", "ModuleX.performance_metric"
    double weight = 1.0;
    std::string goal = "maximize"; // "maximize", "minimize", "target"
    double target_value = 0.0;     // For "target" goal
    double tolerance = 0.01;       // For "target" goal
    bool invert_for_fitness = false; // If a high metric value is bad for fitness
    bool is_primary = true;        // Primary objectives contribute to main fitness
    // bool is_secondary_objective = false; // If for MOEA
};

struct FitnessLandscapeConfigCpp {
    std::string config_id = "flc_cpp_default";
    std::string description = "Default C++ Fitness Landscape";
    std::vector<ObjectiveDefinitionCpp> objective_definitions;
    // std::vector<ConstraintDefinitionCpp> constraints; // Future
    double novelty_search_weight = 0.0; // Weight for novelty in fitness calculation
    double creation_timestamp = 0.0;
    std::string source_directive_sim; // e.g., "creator_request_xyz"

    FitnessLandscapeConfigCpp() = default; // Default constructor
};
// eane_cpp_modules/self_evolution_module/self_evolution_module.h
#pragma once

#include "../core_interface.h"
#include "sem_types.h" // For IndividualCpp, FitnessLandscapeConfigCpp
#include <vector>
#include <string>
#include <deque>    // For novelty archive
#include <random>   // For random operations in GA

class SelfEvolutionModule {
public:
    SelfEvolutionModule(CoreInterface* core,
                        int population_size = 20,
                        double mutation_rate_base = 0.1,
                        double crossover_rate = 0.7,
                        int novelty_archive_size = 100,
                        int novelty_k_neighbors = 15,
                        int abstract_genome_dim = 50);

    void update_logic();

    // --- Public API (Conceptual, might be called from Python or other C++ modules) ---
    bool set_active_fitness_landscape(const FitnessLandscapeConfigCpp& new_landscape_config);
    std::optional<FitnessLandscapeConfigCpp> get_active_fitness_landscape() const;
    // Could add methods to get current population stats, etc.

    // --- For Pybind Access ---
    double get_best_fitness() const;
    double get_average_fitness() const;
    double get_average_novelty() const;
    int get_current_generation_count() const;
    int get_population_size_current() const; // Actual current size

private:
    CoreInterface* core_recombinator_;
    int population_size_target_; // Target size
    double mutation_rate_base_;
    double crossover_rate_;
    int abstract_genome_dim_;

    std::vector<IndividualCpp> current_population_;
    FitnessLandscapeConfigCpp active_fitness_landscape_;
    
    // Novelty Search
    std::deque<Eigen::VectorXd> novelty_archive_; // Stores genomes of novel individuals
    int novelty_archive_max_size_;
    int novelty_k_neighbors_;

    // Internal State (mirroring Python module_state)
    double best_fitness_so_far_current_landscape_ = -1e38;
    double average_fitness_population_ = 0.0;
    double average_novelty_population_ = 0.0;
    int generations_completed_current_landscape_ = 0;
    int stagnation_counter_generations_ = 0;
    // std::string last_best_individual_id_conceptual_; // Handled by IndividualCpp.id

    // Evolutionary Algorithm Steps
    void initialize_population_internal(); // Renamed
    void evaluate_population_internal();   // Renamed
    std::vector<IndividualCpp> select_parents_tournament_internal(int tournament_size = 3); // Renamed
    std::pair<IndividualCpp, IndividualCpp> crossover_blx_alpha_internal( // Renamed
        const IndividualCpp& parent1, const IndividualCpp& parent2, double alpha = 0.5);
    void mutate_genome_gaussian_internal(IndividualCpp& individual, double mutation_strength_factor = 0.1); // Renamed

    // Fitness and Novelty Calculation
    // Returns: <fitness, map_of_secondary_scores, novelty_score>
    std::tuple<double, std::map<std::string, double>, double> evaluate_individual_fitness(IndividualCpp& individual);
    void update_novelty_archive(const IndividualCpp& individual);

    // Interaction with MuGen (Conceptual)
    void propose_best_individual_as_mutation(const IndividualCpp& individual);

    // Helper to initialize default landscape
    void initialize_default_fitness_landscape_internal(); // Renamed

    // Random number generation
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uniform_dist_01_;
    std::normal_distribution<double> normal_dist_std_;
};
// eane_cpp_modules/self_evolution_module/self_evolution_module.cpp
#include "self_evolution_module.h"
#include <algorithm> // std::sort, std::min, std::max
#include <numeric>   // std::accumulate
#include <iostream>  // For logging if core_ is null

SelfEvolutionModule::SelfEvolutionModule(CoreInterface* core,
                                         int population_size,
                                         double mutation_rate_base,
                                         double crossover_rate,
                                         int novelty_archive_size,
                                         int novelty_k_neighbors,
                                         int abstract_genome_dim)
    : core_recombinator_(core),
      population_size_target_(population_size),
      mutation_rate_base_(mutation_rate_base),
      crossover_rate_(crossover_rate),
      abstract_genome_dim_(abstract_genome_dim),
      novelty_archive_max_size_(novelty_archive_size),
      novelty_k_neighbors_(novelty_k_neighbors),
      rng_(std::random_device{}()),
      uniform_dist_01_(0.0, 1.0),
      normal_dist_std_(0.0, 1.0) {

    initialize_default_fitness_landscape_internal();
    initialize_population_internal();

    if (core_recombinator_) {
        core_recombinator_->log_message("INFO", "SelfEvolutionModuleCpp",
            "SelfEvolutionModule C++ (SEM V2.0 DynFit/Novelty - Stub) initialized. Pop Target: " +
            std::to_string(population_size_target_) + ", Genome Dim: " + std::to_string(abstract_genome_dim_));
    }
}

void SelfEvolutionModule::initialize_default_fitness_landscape_internal() {
    active_fitness_landscape_.config_id = "FL_cpp_default_v1";
    active_fitness_landscape_.description = "C++ Default Fitness Landscape: Stability & Core Functionality";
    active_fitness_landscape_.novelty_search_weight = 0.10; // Moderate novelty initially
    active_fitness_landscape_.creation_timestamp = core_recombinator_ ? core_recombinator_->get_current_timestamp() : 0.0;

    active_fitness_landscape_.objective_definitions = {
        {"gs.coherence_score", 0.25, "maximize", 0.0, 0.0, false, true},
        {"gs.system_entropy", 0.20, "target", 0.15, 0.05, true, true}, // Invert: low entropy (around target) = better
        {"gs.phi_funcional_score", 0.20, "maximize", 0.0, 0.0, false, true},
        {"gs.self_esteem", 0.10, "maximize", 0.0, 0.0, false, false},
        {"gs.dolor", 0.15, "minimize", 0.0, 0.0, false, true}, // Minimize implicitly handles inversion for fitness
        // Placeholder for a module health metric. This path needs to be resolvable by get_gs_metric_value_sem.
        // {"GlobalPerformance.avg_module_health_sim_cpp", 0.10, "maximize", 0.0, 0.0, false, false}
    };
}

void SelfEvolutionModule::initialize_population_internal() {
    current_population_.clear();
    current_population_.reserve(population_size_target_);
    for (int i = 0; i < population_size_target_; ++i) {
        Eigen::VectorXd params(abstract_genome_dim_);
        for (int j = 0; j < abstract_genome_dim_; ++j) {
            params(j) = uniform_dist_01_(rng_); // Genomes in [0,1]
        }
        current_population_.emplace_back(params); // Uses constructor IndividualCpp(params)
        current_population_.back().id = "sem_init_cpp_" + std::to_string(i);
    }
    if (core_recombinator_) core_recombinator_->log_message("INFO", "SelfEvolutionModuleCpp", "Initial population of " + std::to_string(current_population_.size()) + " created.");
    
    // Reset generation-specific counters
    best_fitness_so_far_current_landscape_ = -1e38;
    average_fitness_population_ = 0.0;
    average_novelty_population_ = 0.0;
    generations_completed_current_landscape_ = 0;
    stagnation_counter_generations_ = 0;
}

void SelfEvolutionModule::update_logic() {
    if (!core_recombinator_) return;

    // STUB: Handle fitness landscape update events
    // EventDataCpp landscape_event;
    // if (core_recombinator_->event_queue_get_specific_cpp("sem_update_fitness_landscape_config_cpp", landscape_event, 0.001)){
    //    // ... parse landscape_event.content and call set_active_fitness_landscape ...
    // }

    if (current_population_.empty()) {
        initialize_population_internal();
        if (current_population_.empty()) { // Still empty after init, something is wrong
             core_recombinator_->log_message("ERROR", "SelfEvolutionModuleCpp", "Population failed to initialize. Aborting update logic.");
            return;
        }
    }

    // --- Evolve one generation ---
    evaluate_population_internal(); // Calculates fitness and novelty

    // Log population stats
    if (!current_population_.empty()) {
        double sum_fitness = 0.0, sum_novelty = 0.0;
        double current_max_fitness_this_gen = -1e38; // Reset for this generation
        for (const auto& ind : current_population_) {
            if (ind.fitness > -1e37) { // Check if evaluated
                 sum_fitness += ind.fitness;
                 current_max_fitness_this_gen = std::max(current_max_fitness_this_gen, ind.fitness);
            }
            sum_novelty += ind.novelty_score;
        }
        int evaluated_count = 0;
        for(const auto& ind : current_population_) if(ind.fitness > -1e37) evaluated_count++;

        average_fitness_population_ = (evaluated_count > 0) ? sum_fitness / evaluated_count : 0.0;
        average_novelty_population_ = current_population_.empty() ? 0.0 : sum_novelty / current_population_.size();

        if (current_max_fitness_this_gen > best_fitness_so_far_current_landscape_) {
            best_fitness_so_far_current_landscape_ = current_max_fitness_this_gen;
            stagnation_counter_generations_ = 0;
            // Find and propose best individual
            auto best_iter = std::max_element(current_population_.begin(), current_population_.end(),
                                             [](const IndividualCpp& a, const IndividualCpp& b){
                                                 return a.fitness < b.fitness;
                                             });
            if (best_iter != current_population_.end()) {
                 propose_best_individual_as_mutation(*best_iter);
            }
        } else {
            stagnation_counter_generations_++;
        }
    }

    // Selection, Crossover, Mutation to create new population
    std::vector<IndividualCpp> parents = select_parents_tournament_internal();
    std::vector<IndividualCpp> next_generation;
    next_generation.reserve(population_size_target_);

    // Elitism: copy best individuals directly
    int elites_count = std::max(1, static_cast<int>(0.1 * current_population_.size()));
    std::sort(current_population_.begin(), current_population_.end(), [](const IndividualCpp& a, const IndividualCpp& b){
        return a.fitness > b.fitness; // Higher fitness is better
    });
    for (int i = 0; i < elites_count && i < static_cast<int>(current_population_.size()); ++i) {
        next_generation.push_back(current_population_[i]); // Deep copy made by vector push_back
        next_generation.back().age_generations++; // Elite ages
    }

    // Generate offspring
    if (!parents.empty()) {
        std::uniform_int_distribution<int> parent_idx_dist(0, static_cast<int>(parents.size()) - 1);
        while (next_generation.size() < static_cast<size_t>(population_size_target_)) {
            const IndividualCpp& p1 = parents[parent_idx_dist(rng_)];
            const IndividualCpp& p2 = parents[parent_idx_dist(rng_)];

            if (uniform_dist_01_(rng_) < crossover_rate_) {
                auto [child1, child2] = crossover_blx_alpha_internal(p1, p2);
                mutate_genome_gaussian_internal(child1);
                mutate_genome_gaussian_internal(child2);
                next_generation.push_back(child1);
                if (next_generation.size() < static_cast<size_t>(population_size_target_)) {
                    next_generation.push_back(child2);
                }
            } else { // No crossover, pass parents (or copies) with mutation
                IndividualCpp offspring1 = p1; // Copy
                IndividualCpp offspring2 = p2; // Copy
                offspring1.age_generations = 0; // Reset age for new offspring concept
                offspring2.age_generations = 0;
                offspring1.fitness = -1e38; // Mark for re-evaluation
                offspring2.fitness = -1e38;

                mutate_genome_gaussian_internal(offspring1);
                mutate_genome_gaussian_internal(offspring2);
                next_generation.push_back(offspring1);
                if (next_generation.size() < static_cast<size_t>(population_size_target_)) {
                    next_generation.push_back(offspring2);
                }
            }
        }
    } else if (next_generation.empty() && population_size_target_ > 0) {
        // If no parents and no elites, re-initialize to avoid empty population
        core_recombinator_->log_message("WARNING", "SelfEvolutionModuleCpp", "No parents selected and no elites, re-initializing population segment.");
        while (next_generation.size() < static_cast<size_t>(population_size_target_)){
            Eigen::VectorXd params(abstract_genome_dim_);
            for (int j = 0; j < abstract_genome_dim_; ++j) params(j) = uniform_dist_01_(rng_);
            next_generation.emplace_back(params);
        }
    }


    current_population_ = next_generation;
    generations_completed_current_landscape_++;

    // STUB: Check for stagnation and request self-improvement
    // if (stagnation_counter_generations_ > 75 /* Some threshold */) {
    //     // ... send self-improvement request event ...
    //     stagnation_counter_generations_ = 0;
    // }
}

// ... (Implementation of SEM methods will continue in the next part) ...
// eane_cpp_modules/self_evolution_module/self_evolution_module.cpp
// (Continuation from Part 7)
#include "self_evolution_module.h" // Already included

void SelfEvolutionModule::evaluate_population_internal() {
    if (!core_recombinator_) return;
    if (current_population_.empty()) return;

    if (active_fitness_landscape_.objective_definitions.empty()){
        core_recombinator_->log_message("WARNING", "SelfEvolutionModuleCpp", "Evaluation skipped: No active fitness objectives.");
        for(auto& ind : current_population_) {
            ind.fitness = 0.0; // Assign a neutral fitness if no objectives
            ind.novelty_score = 0.0;
        }
        return;
    }

    for (IndividualCpp& ind : current_population_) {
        if (ind.fitness <= -1e37) { // Only evaluate if not already evaluated (or marked as error)
            auto [fit, sec_scores, nov] = evaluate_individual_fitness(ind);
            ind.fitness = fit;
            ind.secondary_objectives_scores = sec_scores; // Store secondary scores
            ind.novelty_score = nov;
            
            if (active_fitness_landscape_.novelty_search_weight > 0) {
                update_novelty_archive(ind);
            }
        }
    }
}

std::vector<IndividualCpp> SelfEvolutionModule::select_parents_tournament_internal(int tournament_size) {
    std::vector<IndividualCpp> selected_parents;
    if (current_population_.empty()) return selected_parents;

    // Ensure tournament size is not larger than population
    tournament_size = std::min(tournament_size, static_cast<int>(current_population_.size()));
    if (tournament_size <= 0) return selected_parents; // Cannot select if tournament size is 0 or less

    selected_parents.reserve(population_size_target_); // Reserve space
    std::uniform_int_distribution<int> idx_dist(0, static_cast<int>(current_population_.size()) - 1);

    for (int i = 0; i < population_size_target_; ++i) {
        IndividualCpp const* winner = nullptr; // Pointer to avoid copying full individuals in tournament
        double best_fitness_in_tournament = -1e38;

        for (int j = 0; j < tournament_size; ++j) {
            const IndividualCpp& participant = current_population_[idx_dist(rng_)];
            if (!winner || participant.fitness > best_fitness_in_tournament) {
                winner = &participant;
                best_fitness_in_tournament = participant.fitness;
            }
        }
        if (winner) {
            selected_parents.push_back(*winner); // Copy the winner
        } else if (!current_population_.empty()) {
            // Fallback if something went wrong (should not happen if tournament_size > 0 and pop not empty)
            selected_parents.push_back(current_population_[0]);
        }
    }
    return selected_parents;
}

std::pair<IndividualCpp, IndividualCpp> SelfEvolutionModule::crossover_blx_alpha_internal(
    const IndividualCpp& parent1, const IndividualCpp& parent2, double alpha) {

    if (parent1.parameters.size() != abstract_genome_dim_ || parent2.parameters.size() != abstract_genome_dim_) {
        // This should not happen if population is initialized correctly
        if(core_recombinator_) core_recombinator_->log_message("ERROR", "SelfEvolutionModuleCpp", "Parent genome dimension mismatch for crossover.");
        return {parent1, parent2}; // Return parents unchanged
    }

    Eigen::VectorXd child1_params(abstract_genome_dim_);
    Eigen::VectorXd child2_params(abstract_genome_dim_);

    for (int i = 0; i < abstract_genome_dim_; ++i) {
        double p1_gene = parent1.parameters(i);
        double p2_gene = parent2.parameters(i);
        double diff = std::abs(p1_gene - p2_gene);
        double min_gene = std::min(p1_gene, p2_gene);
        // double max_gene = std::max(p1_gene, p2_gene); // Not used in BLX-alpha with this formulation

        // Range for BLX-alpha: [min_gene - alpha*diff, max_gene + alpha*diff]
        // Which is [min_gene - alpha*diff, min_gene + diff + alpha*diff]
        // Length of interval = diff + 2*alpha*diff = diff*(1 + 2*alpha)
        double lower_bound = min_gene - alpha * diff;
        double upper_bound = min_gene + diff + alpha * diff; // Equivalent to max_gene + alpha * diff

        std::uniform_real_distribution<double> gene_dist(lower_bound, upper_bound);
        
        child1_params(i) = std::clamp(gene_dist(rng_), 0.0, 1.0); // Clamp to [0,1] for abstract genome
        child2_params(i) = std::clamp(gene_dist(rng_), 0.0, 1.0);
    }
    IndividualCpp child1(child1_params); // New ID will be generated
    IndividualCpp child2(child2_params);
    child1.parent_ids_sim = {parent1.id, parent2.id}; // Store parent IDs
    child2.parent_ids_sim = {parent1.id, parent2.id};
    return {child1, child2};
}

void SelfEvolutionModule::mutate_genome_gaussian_internal(IndividualCpp& individual, double mutation_strength_factor) {
    // Determine adaptive mutation rate
    double adaptive_mutation_rate = mutation_rate_base_;
    if (stagnation_counter_generations_ > 10) { // Example: increase rate if stagnated
        adaptive_mutation_rate = std::min(mutation_rate_base_ * (1.0 + static_cast<double>(stagnation_counter_generations_ - 10) * 0.05), mutation_rate_base_ * 2.5);
    }

    for (int i = 0; i < individual.parameters.size(); ++i) {
        if (uniform_dist_01_(rng_) < adaptive_mutation_rate) {
            double mutation = normal_dist_std_(rng_) * mutation_strength_factor; // Gaussian perturbation
            individual.parameters(i) = std::clamp(individual.parameters(i) + mutation, 0.0, 1.0);
            // Log applied mutation conceptually
            // individual.mutation_ids_applied_sim.push_back("gaussian_gene_" + std::to_string(i));
        }
    }
}

void SelfEvolutionModule::update_novelty_archive(const IndividualCpp& individual) {
    // Add to archive only if it's "sufficiently novel" compared to the archive itself,
    // or if the archive is not full. This prevents the archive from being flooded
    // by very similar individuals if novelty_k_neighbors is small or population is dense.
    // For this stub, we'll keep it simple: add if archive not full, or replace LRU if full.
    
    if (novelty_archive_.size() >= static_cast<size_t>(novelty_archive_max_size_)) {
        novelty_archive_.pop_back(); // Remove least recently added (if using deque as LRU for novelty)
    }
    novelty_archive_.push_front(individual.parameters); // Add most recent novel one to front
}

// --- Public API Methods (Stubs/Simple Implementations) ---
bool SelfEvolutionModule::set_active_fitness_landscape(const FitnessLandscapeConfigCpp& new_landscape_config) {
    if (!core_recombinator_) return false;
    if (new_landscape_config.objective_definitions.empty()) {
        core_recombinator_->log_message("WARNING", "SelfEvolutionModuleCpp", "Attempt to set fitness landscape with no objectives.");
        return false;
    }
    active_fitness_landscape_ = new_landscape_config;
    active_fitness_landscape_.creation_timestamp = core_recombinator_->get_current_timestamp();
    
    // Reset population and generation counters for the new landscape
    initialize_population_internal(); 
    generations_completed_current_landscape_ = 0;
    best_fitness_so_far_current_landscape_ = -1e38;
    stagnation_counter_generations_ = 0;
    novelty_archive_.clear(); // Clear novelty archive for new landscape context

    core_recombinator_->log_message("INFO", "SelfEvolutionModuleCpp", "New fitness landscape '" + new_landscape_config.config_id + "' activated.");
    return true;
}

std::optional<FitnessLandscapeConfigCpp> SelfEvolutionModule::get_active_fitness_landscape() const {
    if (active_fitness_landscape_.objective_definitions.empty()) {
        return std::nullopt; // No valid landscape set
    }
    return active_fitness_landscape_;
}

// --- Accessors for Pybind ---
double SelfEvolutionModule::get_best_fitness() const { return best_fitness_so_far_current_landscape_; }
double SelfEvolutionModule::get_average_fitness() const { return average_fitness_population_; }
double SelfEvolutionModule::get_average_novelty() const { return average_novelty_population_; }
int SelfEvolutionModule::get_current_generation_count() const { return generations_completed_current_landscape_; }
int SelfEvolutionModule::get_population_size_current() const { return static_cast<int>(current_population_.size()); }
// eane_cpp_modules/freewill_module/freewill_types.h
#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>

struct DecisionOptionCpp {
    int id = 0;
    Eigen::VectorXd features; // Feature vector describing the option
    double value_score = 0.0;  // Score based on alignment with EANE's values
    double goal_score = 0.0;   // Score based on alignment with current goals

    explicit DecisionOptionCpp(int option_id = 0, int feature_dim = 0) 
        : id(option_id), features(Eigen::VectorXd::Zero(feature_dim)) {}
};
// eane_cpp_modules/freewill_module/freewill_module.h
#pragma once

#include "../core_interface.h"
#include "freewill_types.h" // For DecisionOptionCpp
#include <vector>
#include <string>
#include <random> // For Gumbel noise and random choices

class FreeWillModule {
public:
    FreeWillModule(CoreInterface* core,
                   int num_options = 10,
                   int feature_dim = 5,
                   double beta = 5.0,   // Softmax temperature for decision probability
                   double sigma = 0.1); // Gumbel noise scale for exploration

    void update_logic();

    // --- For Pybind Access or Internal C++ Use ---
    std::vector<double> get_last_probabilities() const; // Get probabilities of last generated options
    // Potentially expose generated options if needed
    // std::vector<DecisionOptionCpp> get_last_options_generated() const;

private:
    CoreInterface* core_recombinator_;
    int num_options_fw_;     // Number of options to generate each cycle
    int feature_dim_fw_;     // Dimensionality of option features
    double beta_fw_;         // Controls determinism of choice (higher beta = more deterministic)
    double sigma_fw_;        // Controls magnitude of Gumbel noise for exploration

    Eigen::VectorXd value_weights_fw_; // Weights for scoring options against EANE's values
    Eigen::VectorXd goal_weights_fw_;  // Weights for scoring options against current goals

    // Internal state for last cycle (primarily for inspection/Pybind)
    std::vector<DecisionOptionCpp> last_options_generated_cache_;
    std::vector<double> last_probabilities_vector_sample_; // Cached probabilities

    // Helper methods
    std::vector<DecisionOptionCpp> generate_options_fw();
    Eigen::VectorXd compute_value_scores_fw(const std::vector<DecisionOptionCpp>& options);
    Eigen::VectorXd compute_goal_scores_fw(const std::vector<DecisionOptionCpp>& options);
    Eigen::VectorXd introduce_gumbel_noise_fw(int num_elements) const;
    Eigen::VectorXd calculate_selection_probabilities_fw(const Eigen::VectorXd& value_scores, 
                                                         const Eigen::VectorXd& goal_scores);
    double compute_decision_entropy_fw(const Eigen::VectorXd& probabilities) const; // Uses MTK if available

    // Random number generation
    mutable std::mt19937 rng_fw_; // Mutable for const methods that might generate random numbers internally (like Gumbel)
    mutable std::uniform_real_distribution<double> uniform_01_fw_;
};
// eane_cpp_modules/freewill_module/freewill_module.cpp
#include "freewill_module.h"
#include <cmath>     // std::log, std::exp, std::clamp
#include <numeric>   // For std::accumulate for sum if needed
#include <algorithm> // For std::max_element if needed

FreeWillModule::FreeWillModule(CoreInterface* core, int num_options, int feature_dim,
                               double beta, double sigma)
    : core_recombinator_(core), num_options_fw_(num_options), feature_dim_fw_(feature_dim),
      beta_fw_(beta), sigma_fw_(sigma),
      rng_fw_(std::random_device{}()), uniform_01_fw_(0.0, 1.0) {

    if (feature_dim_fw_ > 0) {
        value_weights_fw_ = Eigen::VectorXd::Random(feature_dim_fw_).array().abs(); // Ensure positive weights initially
        double val_sum = value_weights_fw_.sum();
        if (val_sum > 1e-9) value_weights_fw_ /= val_sum; // Normalize
        else value_weights_fw_ = Eigen::VectorXd::Constant(feature_dim_fw_, 1.0 / static_cast<double>(feature_dim_fw_));

        goal_weights_fw_ = Eigen::VectorXd::Random(feature_dim_fw_).array().abs();
        double goal_sum = goal_weights_fw_.sum();
        if (goal_sum > 1e-9) goal_weights_fw_ /= goal_sum;
        else goal_weights_fw_ = Eigen::VectorXd::Constant(feature_dim_fw_, 1.0 / static_cast<double>(feature_dim_fw_));
    } else {
        feature_dim_fw_ = 0; // Ensure it's 0 if invalid
        value_weights_fw_.resize(0);
        goal_weights_fw_.resize(0);
    }
    
    last_options_generated_cache_.reserve(num_options_fw_ > 0 ? num_options_fw_ : 10); // Pre-reserve
    last_probabilities_vector_sample_.reserve(num_options_fw_ > 0 ? num_options_fw_ : 10);

    if(core_recombinator_) core_recombinator_->log_message("INFO", "FreeWillModuleCpp", 
        "FreeWillModule C++ initialized. Options: " + std::to_string(num_options_fw_) + 
        ", FeatureDim: " + std::to_string(feature_dim_fw_));
}

void FreeWillModule::update_logic() {
    if (!core_recombinator_ || feature_dim_fw_ <= 0 || num_options_fw_ <= 0) {
        last_options_generated_cache_.clear();
        last_probabilities_vector_sample_.clear();
        return;
    }

    last_options_generated_cache_ = generate_options_fw();
    if (last_options_generated_cache_.empty()) {
        last_probabilities_vector_sample_.clear();
        return;
    }

    Eigen::VectorXd value_scores_vec = compute_value_scores_fw(last_options_generated_cache_);
    Eigen::VectorXd goal_scores_vec = compute_goal_scores_fw(last_options_generated_cache_);

    // Update scores in the cached options (for potential inspection)
    for (size_t i = 0; i < last_options_generated_cache_.size(); ++i) {
        if (i < static_cast<size_t>(value_scores_vec.size())) 
            last_options_generated_cache_[i].value_score = value_scores_vec(i);
        if (i < static_cast<size_t>(goal_scores_vec.size())) 
            last_options_generated_cache_[i].goal_score = goal_scores_vec(i);
    }
    
    Eigen::VectorXd probabilities_vec = calculate_selection_probabilities_fw(value_scores_vec, goal_scores_vec);
    // double decision_entropy = compute_decision_entropy_fw(probabilities_vec); // Optional to calculate

    // Store probabilities for Pybind access
    last_probabilities_vector_sample_.assign(probabilities_vec.data(), probabilities_vec.data() + probabilities_vec.size());

    // Send event to FreeWillEngine (Python or C++)
    EventDataCpp fwe_input_event("free_will_options_generated_for_engine_cpp"); // Type for C++ FWE
    fwe_input_event.priority_label = "medium";
    fwe_input_event.source_module = "FreeWillModuleCpp";

    // Content: Need to serialize options and probabilities if EventContentValueCpp doesn't handle them directly.
    // For a Python FWE, the pybind wrapper would handle this.
    // For a C++ FWE, it might expect specific types in the variant.
    // Simplification: send counts and a sample for now.
    // A more robust way is to define structs for event content or use JSON strings.
    fwe_input_event.content["options_count_val"] = static_cast<int>(last_options_generated_cache_.size());
    if (!last_options_generated_cache_.empty()) {
         fwe_input_event.content["first_option_id_val"] = last_options_generated_cache_[0].id;
    }
    // To pass full data to a C++ FWE, one might use std::vector<DecisionOptionCpp> in the variant,
    // or pass pointers/references if lifetime is managed carefully (not recommended across event queue).
    // For passing to Python FWE, this C++ module doesn't construct the Python-compatible event directly;
    // the pybind wrapper for *this module's output* would create the Python dict.

    // If FWE is C++, it might register for a "native_fwm_options_available" event
    // and then call a method on this FWM instance to get last_options_generated_cache_ and last_probabilities_vector_sample_
    // This avoids complex serialization through the generic event queue.

    core_recombinator_->event_queue_put(fwe_input_event);
}

std::vector<DecisionOptionCpp> FreeWillModule::generate_options_fw() {
    std::vector<DecisionOptionCpp> options;
    options.reserve(num_options_fw_);
    for (int i = 0; i < num_options_fw_; ++i) {
        options.emplace_back(i, feature_dim_fw_); // ID i, features of feature_dim_fw_
        // Populate features with random values (example: N(0, 0.5))
        for (int j = 0; j < feature_dim_fw_; ++j) {
            options.back().features(j) = normal_dist_std_(rng_fw_) * 0.5;
        }
    }
    return options;
}

Eigen::VectorXd FreeWillModule::compute_value_scores_fw(const std::vector<DecisionOptionCpp>& options) {
    if (options.empty() || value_weights_fw_.size() != feature_dim_fw_) return Eigen::VectorXd(0);
    Eigen::VectorXd scores(options.size());
    for (size_t i = 0; i < options.size(); ++i) {
        if (options[i].features.size() == feature_dim_fw_) {
            scores(i) = value_weights_fw_.dot(options[i].features);
        } else { scores(i) = 0; } // Mismatch
    }
    // Normalize scores (e.g., to [-1, 1] if dot products can be large)
    double max_abs_score = scores.array().abs().maxCoeff();
    if (max_abs_score > 1e-9) scores /= max_abs_score;
    return scores;
}

Eigen::VectorXd FreeWillModule::compute_goal_scores_fw(const std::vector<DecisionOptionCpp>& options) {
    if (options.empty() || goal_weights_fw_.size() != feature_dim_fw_) return Eigen::VectorXd(0);
    Eigen::VectorXd scores(options.size());
    for (size_t i = 0; i < options.size(); ++i) {
         if (options[i].features.size() == feature_dim_fw_) {
            scores(i) = goal_weights_fw_.dot(options[i].features);
        } else { scores(i) = 0; }
    }
    double max_abs_score = scores.array().abs().maxCoeff();
    if (max_abs_score > 1e-9) scores /= max_abs_score;
    return scores;
}

Eigen::VectorXd FreeWillModule::introduce_gumbel_noise_fw(int num_elements) const {
    if (num_elements <= 0) return Eigen::VectorXd(0);
    Eigen::VectorXd gumbel_noise(num_elements);
    for (int i = 0; i < num_elements; ++i) {
        double u = uniform_01_fw_(rng_fw_);
        // Clamp u to avoid log(0) or log(log(0))
        u = std::clamp(u, 1e-9, 1.0 - 1e-9); 
        gumbel_noise(i) = -std::log(-std::log(u)); // Standard Gumbel
    }
    return gumbel_noise * sigma_fw_; // Scale by sigma
}

Eigen::VectorXd FreeWillModule::calculate_selection_probabilities_fw(
    const Eigen::VectorXd& value_scores, const Eigen::VectorXd& goal_scores) {
    
    if (value_scores.size() != goal_scores.size() || value_scores.size() == 0) {
        // Return uniform if sizes mismatch or empty
        int n_options = static_cast<int>(std::max(value_scores.size(), goal_scores.size()));
        if (n_options == 0) n_options = num_options_fw_; // Fallback
        if (n_options == 0) return Eigen::VectorXd(0);
        return Eigen::VectorXd::Constant(n_options, 1.0 / static_cast<double>(n_options));
    }

    Eigen::VectorXd combined_scores = value_scores + goal_scores; // Simple sum for now
    Eigen::VectorXd gumbel_noise = introduce_gumbel_noise_fw(static_cast<int>(combined_scores.size()));
    Eigen::VectorXd logits = beta_fw_ * (combined_scores + gumbel_noise);

    // Softmax for probabilities (stabilized)
    Eigen::VectorXd exp_logits = (logits.array() - logits.maxCoeff()).exp(); // Subtract max for numerical stability
    double sum_exp_logits = exp_logits.sum();

    if (sum_exp_logits > 1e-10) {
        return exp_logits / sum_exp_logits;
    } else {
        // Fallback to uniform if sum is too small (e.g., all logits very negative)
        return Eigen::VectorXd::Constant(logits.size(), 1.0 / static_cast<double>(logits.size()));
    }
}

double FreeWillModule::compute_decision_entropy_fw(const Eigen::VectorXd& probabilities) const {
    if (!core_recombinator_) return 0.0; // Cannot access MTK
    // This would ideally use mtk_->calculate_shannon_entropy(probabilities);
    // Fallback manual calculation if MTK is not available or method not exposed:
    double entropy = 0.0;
    for (int i = 0; i < probabilities.size(); ++i) {
        double p = probabilities(i);
        if (p > 1e-12) {
            entropy -= p * std::log2(p);
        }
    }
    return entropy;
}

// --- Accessors for Pybind ---
std::vector<double> FreeWillModule::get_last_probabilities() const {
    return last_probabilities_vector_sample_;
}
// eane_cpp_modules/freewill_engine/environment_fwe.h
#pragma once
#include <Eigen/Dense>
#include <random>   // For std::mt19937, std::normal_distribution
#include <utility>  // For std::pair

class EnvironmentFWE {
public:
    EnvironmentFWE(int state_dim = 5, int num_actions = 10);

    Eigen::VectorXd get_current_state_env() const; // Made const, internal state modified if needed
    
    // Applies action, updates internal state, returns <next_state, reward>
    std::pair<Eigen::VectorXd, double> apply_action_to_env(int action_id);

private:
    int state_dim_env_;
    int num_actions_env_;
    Eigen::VectorXd current_env_state_; // The state of the simulated environment

    // Random number generation for environment dynamics
    mutable std::mt19937 rng_env_; // Mutable for use in const get_current_state_env if it adds noise for observation
    mutable std::normal_distribution<double> noise_dist_env_; // For state transitions or observation noise
    mutable std::normal_distribution<double> reward_noise_dist_env_; // For reward noise
};
// eane_cpp_modules/freewill_engine/environment_fwe.cpp
#include "environment_fwe.h"
#include <algorithm> // std::clamp

EnvironmentFWE::EnvironmentFWE(int state_dim, int num_actions)
    : state_dim_env_(std::max(0, state_dim)), 
      num_actions_env_(std::max(0, num_actions)),
      rng_env_(std::random_device{}()),
      noise_dist_env_(0.0, 0.02), // Small noise for state transitions/observation
      reward_noise_dist_env_(0.0, 0.1) // Noise for base reward
{
    if (state_dim_env_ > 0) {
        current_env_state_ = Eigen::VectorXd::Random(state_dim_env_) * 0.2; // Initial state near origin
    } else {
        current_env_state_.resize(0);
    }
}

Eigen::VectorXd EnvironmentFWE::get_current_state_env() const {
    // If observation should be noisy, apply noise here to a copy.
    // For simplicity, returning the true current state.
    // If state itself drifts over time (even without action), update here:
    // for (int i = 0; i < state_dim_env_; ++i) {
    //     current_env_state_(i) += noise_dist_env_(rng_env_); // Apply small drift
    //     current_env_state_(i) = std::clamp(current_env_state_(i), -1.0, 1.0);
    // }
    return current_env_state_; // Returns a copy due to Eigen's copy-on-write or value semantics
}

std::pair<Eigen::VectorXd, double> EnvironmentFWE::apply_action_to_env(int action_id) {
    if (state_dim_env_ <= 0 || num_actions_env_ <= 0 || action_id < 0 || action_id >= num_actions_env_) {
        // Invalid action or uninitialized environment
        return {current_env_state_, 0.0}; // Return current state and zero reward
    }

    // Simulate base reward for the action
    double base_reward = reward_noise_dist_env_(rng_env_); // N(0, 0.1)

    // Add some preference for certain actions (example arbitrary logic)
    double action_preference_bonus = 0.0;
    if (action_id % 2 == 0) action_preference_bonus += 0.15;
    if (action_id == (num_actions_env_ / 2)) action_preference_bonus += 0.1; // Middle action preferred
    
    double total_env_reward = base_reward + action_preference_bonus;

    // Simulate action effect on state
    Eigen::VectorXd action_effect = Eigen::VectorXd::Random(state_dim_env_) * 0.05; // Base random effect
    // Make effect somewhat dependent on action ID (conceptual)
    action_effect *= (static_cast<double>(action_id + 1) / static_cast<double>(num_actions_env_));
    
    current_env_state_ += action_effect;
    // Add general small drift/noise to state after action
    for (int i = 0; i < state_dim_env_; ++i) {
        current_env_state_(i) += noise_dist_env_(rng_env_);
        current_env_state_(i) = std::clamp(current_env_state_(i), -1.0, 1.0); // Keep state bounded
    }
    
    return {current_env_state_, std::clamp(total_env_reward, -0.5, 0.5)}; // Clamp reward
}
// eane_cpp_modules/freewill_engine/freewill_engine.h
#pragma once
#include "../core_interface.h"
#include "../freewill_module/freewill_types.h" // For DecisionOptionCpp
#include "environment_fwe.h"       // For the simulated environment
#include <vector>
#include <string>
#include <map>
#include <random>
#include <optional> // C++17

// Key for Q-table: pair of (discretized state vector, action_id)
// Discretized state is represented as std::vector<double> for map compatibility.
struct VectorDoubleKeyCompareFWE { // Renamed to avoid conflict
    bool operator()(const std::vector<double>& a, const std::vector<double>& b) const {
        return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
    }
};
using StateActionKeyFweInternal = std::pair<std::vector<double>, int>; // Renamed

class FreeWillEngine {
public:
    FreeWillEngine(CoreInterface* core, 
                   int num_actions = 10,         // Should match num_options from FreeWillModule
                   int state_dim_env = 5,        // Dimensionality of the environment state FWE interacts with
                   double alpha_lr = 0.1,        // Q-learning learning rate
                   double gamma_discount = 0.9,  // Q-learning discount factor
                   double epsilon_start = 0.8,
                   double epsilon_end = 0.05, 
                   double epsilon_decay_rate = 0.001);

    // Main update logic. In a C++ only system, it might fetch options.
    // For Python integration, this might be called BY Python after FWM runs.
    void update_logic(); 

    // Method for Python to feed options generated by Python's FreeWillModule
    // This is crucial for bridging the Python FWM with C++ FWE.
    void process_freewill_module_output(
        const std::vector<DecisionOptionCpp>& options_from_fwm, // Assumes features match state_dim_env_
        const Eigen::VectorXd& probabilities_from_fwm);


    // --- Accessors for Pybind or internal inspection ---
    std::optional<int> get_last_selected_action_id() const;
    double get_last_total_reward() const;
    int get_q_table_size() const;
    double get_current_epsilon() const;

private:
    CoreInterface* core_recombinator_;
    int num_actions_fwe_;      // Number of discrete actions FWE can choose from (corresponds to FWM options)
    int state_dim_env_fwe_;    // Dimensionality of the environment state FWE perceives for Q-learning
    
    double alpha_lr_fwe_;       // Learning rate
    double gamma_discount_fwe_; // Discount factor
    double epsilon_start_fwe_;
    double epsilon_end_fwe_;
    double epsilon_decay_rate_fwe_;
    long long time_step_counter_fwe_ = 0; // For epsilon decay

    std::map<StateActionKeyFweInternal, double, VectorDoubleKeyCompareFWE> q_table_fwe_;
    EnvironmentFWE simulated_environment_fwe_; // Internal simulated environment for learning

    // Cache for options received from FreeWillModule (if passed via process_freewill_module_output)
    std::optional<std::vector<DecisionOptionCpp>> last_received_options_fwm_;
    std::optional<Eigen::VectorXd> last_received_probabilities_fwm_;

    // Internal state mirroring Python module_state
    std::optional<int> last_selected_action_id_cache_; // Using optional
    double last_total_reward_cache_ = 0.0;

    // Helper methods
    std::vector<double> discretize_state_for_q_table(const Eigen::VectorXd& state_vector) const;
    double get_q_value(const Eigen::VectorXd& state_vector, int action_id) const;
    double calculate_total_reward_internal(int action_id, const DecisionOptionCpp& selected_option, double env_reward) const; // Renamed

    // Selects action: returns <action_id, pointer to selected DecisionOptionCpp object>
    std::pair<int, const DecisionOptionCpp*> select_action_internal( // Renamed
        const Eigen::VectorXd& current_env_state,
        const std::vector<DecisionOptionCpp>& available_options,
        const Eigen::VectorXd& probabilities_fwm);

    void update_q_table_internal(const Eigen::VectorXd& state_vector, int action_id, // Renamed
                                 double reward, const Eigen::VectorXd& next_state_vector);

    // Random number generation
    mutable std::mt19937 rng_fwe_internal_; // Renamed
    mutable std::uniform_real_distribution<double> uniform_01_fwe_internal_; // Renamed
};
// eane_cpp_modules/freewill_engine/freewill_engine.cpp
#include "freewill_engine.h"
#include <cmath>     // std::exp, std::log, std::round
#include <algorithm> // std::max_element, std::transform, std::clamp
#include <limits>    // std::numeric_limits
#include <iostream>  // For logging if core is null

FreeWillEngine::FreeWillEngine(CoreInterface* core, int num_actions, int state_dim_env,
                               double alpha_lr, double gamma_discount, double epsilon_start,
                               double epsilon_end, double epsilon_decay_rate)
    : core_recombinator_(core), num_actions_fwe_(num_actions), state_dim_env_fwe_(state_dim_env),
      alpha_lr_fwe_(alpha_lr), gamma_discount_fwe_(gamma_discount),
      epsilon_start_fwe_(epsilon_start), epsilon_end_fwe_(epsilon_end), epsilon_decay_rate_fwe_(epsilon_decay_rate),
      simulated_environment_fwe_(state_dim_env, num_actions), // Init environment
      rng_fwe_internal_(std::random_device{}()), uniform_01_fwe_internal_(0.0, 1.0) {

    if (num_actions_fwe_ <= 0) {
        // Log error or throw, as FWE cannot operate without actions
        if(core_recombinator_) core_recombinator_->log_message("ERROR", "FreeWillEngineCpp", "num_actions_fwe must be positive.");
        num_actions_fwe_ = 0; // Prevent further issues
    }
    if (state_dim_env_fwe_ <= 0) {
        if(core_recombinator_) core_recombinator_->log_message("ERROR", "FreeWillEngineCpp", "state_dim_env_fwe must be positive.");
        state_dim_env_fwe_ = 0;
    }

    if(core_recombinator_) {
        core_recombinator_->log_message("INFO", "FreeWillEngineCpp", 
            "FreeWillEngine C++ initialized. Actions: " + std::to_string(num_actions_fwe_) + 
            ", EnvStateDim: " + std::to_string(state_dim_env_fwe_));
    }
}

void FreeWillEngine::process_freewill_module_output(
    const std::vector<DecisionOptionCpp>& options_from_fwm,
    const Eigen::VectorXd& probabilities_from_fwm) {
    
    if (options_from_fwm.size() != static_cast<size_t>(num_actions_fwe_)) {
        if(core_recombinator_) core_recombinator_->log_message("WARNING", "FreeWillEngineCpp", 
            "Number of options from FWM (" + std::to_string(options_from_fwm.size()) + 
            ") does not match FWE configured num_actions (" + std::to_string(num_actions_fwe_) + "). Output ignored.");
        last_received_options_fwm_.reset();
        last_received_probabilities_fwm_.reset();
        return;
    }
    if (probabilities_from_fwm.size() != num_actions_fwe_ || 
        (probabilities_from_fwm.size() > 0 && (std::abs(probabilities_from_fwm.sum() - 1.0) > 1e-5 || (probabilities_from_fwm.array() < 0.0).any())) ) {
        if(core_recombinator_) core_recombinator_->log_message("WARNING", "FreeWillEngineCpp", 
            "Probabilities from FWM are invalid (size mismatch, not sum to 1, or negative). Output ignored.");
        last_received_options_fwm_.reset();
        last_received_probabilities_fwm_.reset();
        return;
    }

    last_received_options_fwm_ = options_from_fwm;
    last_received_probabilities_fwm_ = probabilities_from_fwm;
    if(core_recombinator_) core_recombinator_->log_message("DEBUG", "FreeWillEngineCpp", "FWM output processed and cached.");
}


void FreeWillEngine::update_logic() {
    if (!core_recombinator_ || num_actions_fwe_ <= 0 || state_dim_env_fwe_ <= 0) {
        return; // Not operational
    }

    // Check if we have fresh options from FWM (Python side)
    if (!last_received_options_fwm_ || !last_received_probabilities_fwm_) {
        // core_recombinator_->log_message("DEBUG", "FreeWillEngineCpp", "No new options from FWM to process in this cycle.");
        return; // Wait for new options
    }
    
    // Use the cached options and probabilities
    const std::vector<DecisionOptionCpp>& current_options = *last_received_options_fwm_;
    const Eigen::VectorXd& current_probabilities = *last_received_probabilities_fwm_;

    // Clear the cache after processing to ensure new data is used next time
    last_received_options_fwm_.reset();
    last_received_probabilities_fwm_.reset();

    time_step_counter_fwe_++;
    Eigen::VectorXd current_env_state = simulated_environment_fwe_.get_current_state_env();

    auto [selected_action_id, selected_option_ptr] = select_action_internal(current_env_state, current_options, current_probabilities);

    if (!selected_option_ptr || selected_action_id < 0) {
        if(core_recombinator_) core_recombinator_->log_message("ERROR", "FreeWillEngineCpp", "Failed to select a valid action.");
        last_selected_action_id_cache_.reset();
        return;
    }
    last_selected_action_id_cache_ = selected_action_id;

    auto [next_env_state, env_reward] = simulated_environment_fwe_.apply_action_to_env(*last_selected_action_id_cache_);
    last_total_reward_cache_ = calculate_total_reward_internal(*last_selected_action_id_cache_, *selected_option_ptr, env_reward);
    update_q_table_internal(current_env_state, *last_selected_action_id_cache_, last_total_reward_cache_, next_env_state);

    // Update GlobalSelfState (via CoreInterface) and send event
    GlobalSelfStateCpp& gs = core_recombinator_->get_global_state();
    // Using std::map for gs.decision for simplicity in bridging
    gs.last_decision_cpp["action_id_val"] = *last_selected_action_id_cache_;
    gs.last_decision_cpp["reward_val"] = last_total_reward_cache_;
    gs.last_decision_cpp["source_module_str"] = std::string("FreeWillEngineCpp");
    gs.last_decision_cpp["timestamp_val"] = core_recombinator_->get_current_timestamp();
    // Potentially add features of selected option if useful
    // gs.last_decision_cpp["selected_option_features_vec_val"] = selected_option_ptr->features; // Need EventContentValueCpp to handle Eigen

    EventDataCpp decision_event("engine_decision_executed_cpp");
    decision_event.content["action_id_val"] = *last_selected_action_id_cache_;
    decision_event.content["total_reward_achieved_val"] = last_total_reward_cache_;
    // To pass features, ensure EventContentValueCpp can handle Eigen::VectorXd or convert to std::vector<double>
    // std::vector<double> features_vec(selected_option_ptr->features.data(), selected_option_ptr->features.data() + selected_option_ptr->features.size());
    // decision_event.content["selected_option_features_vec_val"] = features_vec;
    decision_event.priority_label = "high";
    core_recombinator_->event_queue_put(decision_event);

    if(core_recombinator_) core_recombinator_->log_message("INFO", "FreeWillEngineCpp", 
        "Decision executed: Action ID " + std::to_string(*last_selected_action_id_cache_) + 
        ", Reward: " + std::to_string(last_total_reward_cache_));
}

double FreeWillEngine::get_current_epsilon() const {
    return epsilon_end_fwe_ +
           (epsilon_start_fwe_ - epsilon_end_fwe_) * std::exp(-epsilon_decay_rate_fwe_ * static_cast<double>(time_step_counter_fwe_));
}

std::vector<double> FreeWillEngine::discretize_state_for_q_table(const Eigen::VectorXd& state_vector) const {
    std::vector<double> discretized_key(state_vector.size());
    for (int i = 0; i < state_vector.size(); ++i) {
        // Round to 1 decimal place to keep state space manageable for a map-based Q-table
        discretized_key[i] = std::round(state_vector(i) * 10.0) / 10.0;
    }
    return discretized_key;
}

double FreeWillEngine::get_q_value(const Eigen::VectorXd& state_vector, int action_id) const {
    if (action_id < 0 || action_id >= num_actions_fwe_) return 0.0; // Invalid action
    StateActionKeyFweInternal key = {discretize_state_for_q_table(state_vector), action_id};
    auto it = q_table_fwe_.find(key);
    return (it != q_table_fwe_.end()) ? it->second : 0.0; // Default Q-value is 0 for unseen state-actions
}

double FreeWillEngine::calculate_total_reward_internal(int action_id, const DecisionOptionCpp& selected_option, double env_reward) const {
    // Reward components (weights can be tuned)
    double reward_from_option_value = 0.4 * selected_option.value_score;
    double reward_from_option_goal = 0.3 * selected_option.goal_score;
    // Exploration bonus decays over time and with lower epsilon
    double exploration_bonus = 0.1 * get_current_epsilon() * (1.0 / (1.0 + static_cast<double>(time_step_counter_fwe_) * 0.005));

    // Weighted sum of reward components
    double w_env = 0.5;
    double w_option = 0.4;
    double w_explore = 0.1; // Small weight for exploration bonus
    
    double total_reward = (w_env * env_reward +
                           w_option * (reward_from_option_value + reward_from_option_goal) +
                           w_explore * exploration_bonus);
    return std::clamp(total_reward, -1.0, 1.0); // Clamp reward to a standard range
}

std::pair<int, const DecisionOptionCpp*> FreeWillEngine::select_action_internal(
    const Eigen::VectorXd& current_env_state,
    const std::vector<DecisionOptionCpp>& available_options,
    const Eigen::VectorXd& probabilities_fwm) {

    if (available_options.empty()) return {-1, nullptr};

    double current_epsilon_val = get_current_epsilon(); // Renamed to avoid conflict
    int selected_option_original_id = -1;
    const DecisionOptionCpp* selected_option_ptr = nullptr;

    if (uniform_01_fwe_internal_(rng_fwe_internal_) < current_epsilon_val) { // Explore
        // Use probabilities from FWM if valid, otherwise uniform random
        if (probabilities_fwm.size() == static_cast<long>(available_options.size()) &&
            std::abs(probabilities_fwm.sum() - 1.0) < 1e-5 &&
            (probabilities_fwm.array() >= 0.0).all()) {
            
            std::discrete_distribution<> dist(probabilities_fwm.data(), probabilities_fwm.data() + probabilities_fwm.size());
            int selected_idx = dist(rng_fwe_internal_);
            selected_option_original_id = available_options[selected_idx].id;
            selected_option_ptr = &available_options[selected_idx];
        } else {
            if(core_recombinator_ && probabilities_fwm.size() > 0) core_recombinator_->log_message("WARNING", "FreeWillEngineCpp", "Exploration: Invalid FWM probabilities. Using uniform random choice.");
            std::uniform_int_distribution<int> uniform_idx_dist(0, static_cast<int>(available_options.size()) - 1);
            int selected_idx = uniform_idx_dist(rng_fwe_internal_);
            selected_option_original_id = available_options[selected_idx].id;
            selected_option_ptr = &available_options[selected_idx];
        }
    } else { // Exploit
        double max_q_val = -std::numeric_limits<double>::infinity();
        std::vector<int> best_action_indices; // Store indices in available_options

        for (size_t i = 0; i < available_options.size(); ++i) {
            // available_options[i].id is the action_id for Q-table
            double q_val = get_q_value(current_env_state, available_options[i].id);
            // Add small Gumbel noise to Q-values for tie-breaking and slight exploration during exploitation
            double u_gumbel = uniform_01_fwe_internal_(rng_fwe_internal_);
            u_gumbel = std::clamp(u_gumbel, 1e-7, 1.0 - 1e-7); // Avoid log(0)
            double gumbel_noise = -std::log(-std::log(u_gumbel)) * 0.001; // Very small noise scale
            q_val += gumbel_noise;

            if (q_val > max_q_val) {
                max_q_val = q_val;
                best_action_indices.clear();
                best_action_indices.push_back(static_cast<int>(i));
            } else if (std::abs(q_val - max_q_val) < 1e-9) { // Tie
                best_action_indices.push_back(static_cast<int>(i));
            }
        }
        
        if (!best_action_indices.empty()) {
            std::uniform_int_distribution<int> tie_breaker_dist(0, static_cast<int>(best_action_indices.size() - 1));
            int selected_idx = best_action_indices[tie_breaker_dist(rng_fwe_internal_)];
            selected_option_original_id = available_options[selected_idx].id;
            selected_option_ptr = &available_options[selected_idx];
        } else {
            // Should not happen if available_options is not empty. Fallback to random.
            std::uniform_int_distribution<int> fallback_dist(0, static_cast<int>(available_options.size() - 1));
            int selected_idx = fallback_dist(rng_fwe_internal_);
            selected_option_original_id = available_options[selected_idx].id;
            selected_option_ptr = &available_options[selected_idx];
        }
    }
    return {selected_option_original_id, selected_option_ptr};
}

void FreeWillEngine::update_q_table_internal(const Eigen::VectorXd& state_vector, int action_id,
                                             double reward, const Eigen::VectorXd& next_state_vector) {
    if (action_id < 0 || action_id >= num_actions_fwe_) return; // Invalid action

    StateActionKeyFweInternal current_sa_key = {discretize_state_for_q_table(state_vector), action_id};
    double old_q_value = q_table_fwe_[current_sa_key]; // operator[] creates if not exists (value-initializes to 0.0 for double)

    double max_next_q_action_value = 0.0;
    if (num_actions_fwe_ > 0) {
        max_next_q_action_value = -std::numeric_limits<double>::infinity();
        for (int next_action = 0; next_action < num_actions_fwe_; ++next_action) {
            max_next_q_action_value = std::max(max_next_q_action_value, get_q_value(next_state_vector, next_action));
        }
        if (max_next_q_action_value == -std::numeric_limits<double>::infinity()){
             max_next_q_action_value = 0.0; // If all next Qs are uninit or -inf
        }
    }

    double new_q_value = old_q_value + alpha_lr_fwe_ * (reward + gamma_discount_fwe_ * max_next_q_action_value - old_q_value);
    q_table_fwe_[current_sa_key] = new_q_value;
}

// --- Accessors for Pybind ---
std::optional<int> FreeWillEngine::get_last_selected_action_id() const { return last_selected_action_id_cache_; }
double FreeWillEngine::get_last_total_reward() const { return last_total_reward_cache_; }
int FreeWillEngine::get_q_table_size() const { return static_cast<int>(q_table_fwe_.size()); }
double FreeWillEngine::get_current_epsilon() const { return get_current_epsilon(); /* Calls the private helper */ }
// eane_cpp_modules/adaptive_firewall_module/firewall_types.cpp
#include "firewall_types.h"
#include <algorithm> // For std::transform for to_lower
#include <iostream>  // For debug output if regex fails

// Helper to convert string to lower for case-insensitive comparisons
std::string to_lower_firewall(std::string s) { // Renamed to avoid conflict
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return s;
}

bool FirewallRuleCpp::compile_payload_regex() {
    if (!payload_regex_str_original.empty()) {
        try {
            payload_regex_compiled = std::regex(payload_regex_str_original, std::regex_constants::icase | std::regex_constants::optimize);
            return true;
        } catch (const std::regex_error& e) {
            // std::cerr << "FirewallRuleCpp Regex Error: Failed to compile payload regex '"
            //           << payload_regex_str_original << "': " << e.what() << std::endl;
            payload_regex_compiled.reset(); // Ensure it's empty if compilation fails
            return false;
        }
    }
    payload_regex_compiled.reset(); // No regex string, so no compiled regex
    return true; // Technically successful as there was nothing to compile or it was empty
}


// Simplified IP matching: exact match or basic prefix for CIDR-like patterns (ending with '/')
// A full CIDR/regex IP matcher is more complex.
bool FirewallRuleCpp::match_ip_cidr_regex_stub(const std::optional<std::string>& pattern_str_opt,
                                               const std::optional<std::string>& ip_to_check_opt) const {
    if (!pattern_str_opt) return true; // No pattern, matches all
    if (!ip_to_check_opt || ip_to_check_opt->empty()) return false; // Pattern exists, but no IP to check

    const std::string& pattern = *pattern_str_opt;
    const std::string& ip_to_check = *ip_to_check_opt;

    if (pattern == "*" || pattern == "any" || pattern.empty()) return true;

    // Basic CIDR-like prefix match (e.g., "192.168.1.0/24" - this stub only handles simple prefix part)
    // A real implementation would parse the prefix length.
    // For simplicity, if pattern ends with '/', treat it as a prefix up to that point.
    // Or, if it's a simple IP, it's an exact match.
    // This stub does not handle complex regex for IPs yet.
    size_t cidr_slash_pos = pattern.find('/');
    if (cidr_slash_pos != std::string::npos) {
        std::string pattern_prefix = pattern.substr(0, cidr_slash_pos);
        // Simple prefix check: does ip_to_check start with pattern_prefix?
        // This is a very basic interpretation and not a full CIDR match.
        if (ip_to_check.rfind(pattern_prefix, 0) == 0) { // starts_with
            // A full CIDR match would also check the prefix length.
            // For now, if the IP address starts with the network part, consider it a match.
            return true;
        }
        return false;
    }
    
    // Exact IP match
    return pattern == ip_to_check;
}


bool FirewallRuleCpp::matches(const std::map<std::string, PacketInfoValueVariant>& packet_info) const {
    if (!enabled) return false;

    // Helper lambdas to safely get values from packet_info map
    auto get_str_val = [&](const std::string& key) -> std::optional<std::string> {
        auto it = packet_info.find(key);
        if (it != packet_info.end() && std::holds_alternative<std::string>(it->second)) {
            return std::get<std::string>(it->second);
        }
        return std::nullopt;
    };
    auto get_int_val = [&](const std::string& key) -> std::optional<int> {
        auto it = packet_info.find(key);
        if (it != packet_info.end() && std::holds_alternative<int>(it->second)) {
            return std::get<int>(it->second);
        }
        return std::nullopt;
    };
    auto get_double_val = [&](const std::string& key) -> std::optional<double> {
        auto it = packet_info.find(key);
        if (it != packet_info.end() && std::holds_alternative<double>(it->second)) {
            return std::get<double>(it->second);
        }
        return std::nullopt;
    };

    if (!match_ip_cidr_regex_stub(src_ip_pattern_str, get_str_val("src_ip"))) return false;
    if (!match_ip_cidr_regex_stub(dst_ip_pattern_str, get_str_val("dst_ip"))) return false;

    std::optional<int> pkt_src_port = get_int_val("src_port");
    if (src_port && (!pkt_src_port || *src_port != *pkt_src_port)) return false;

    std::optional<int> pkt_dst_port = get_int_val("dst_port");
    if (dst_port && (!pkt_dst_port || *dst_port != *pkt_dst_port)) return false;

    std::optional<std::string> pkt_protocol = get_str_val("protocol");
    if (protocol && pkt_protocol) {
        std::string rule_proto_lower = to_lower_firewall(*protocol);
        if (rule_proto_lower != "any" && rule_proto_lower != to_lower_firewall(*pkt_protocol)) {
            return false;
        }
    } else if (protocol && !pkt_protocol && *protocol != "any") { // Rule specifies protocol, packet doesn't, and rule isn't "any"
        return false;
    }

    if (payload_regex_compiled) { // Check if regex is compiled and available
        std::optional<std::string> payload = get_str_val("payload_sample");
        if (!payload || !std::regex_search(*payload, *payload_regex_compiled)) {
            return false;
        }
    }

    if (min_threat_score) {
        std::optional<double> pkt_threat_score = get_double_val("threat_score");
        if (!pkt_threat_score || *pkt_threat_score < *min_threat_score) {
            return false;
        }
    }
    return true; // All conditions passed
}
// eane_cpp_modules/adaptive_firewall_module/adaptive_firewall_module.h
#pragma once
#include "../core_interface.h"
#include "firewall_types.h" // For FirewallRuleCpp, TrafficFeatureVectorCpp
#include <vector>
#include <list>     // For rules, allowing efficient insertion/deletion if sorted by priority
#include <string>
#include <deque>    // For recent_traffic_log_
#include <map>      // For storing model (conceptually)
#include <random>   // For simulating traffic

class AdaptiveFirewallModule {
public:
    explicit AdaptiveFirewallModule(CoreInterface* core);
    void update_logic();

    // --- Rule Management API ---
    std::string add_rule(const FirewallRuleCpp& rule_data); // Returns rule_id or empty if failed
    bool remove_rule(const std::string& rule_id);
    bool update_rule(const std::string& rule_id, const FirewallRuleCpp& new_rule_data);
    std::optional<FirewallRuleCpp> get_rule(const std::string& rule_id) const;
    std::vector<FirewallRuleCpp> get_all_rules() const; // Returns a copy of all rules

    // --- Threat Model API (Stubs) ---
    void train_threat_model_stub(const std::vector<TrafficFeatureVectorCpp>& benign_sample,
                                 const std::vector<TrafficFeatureVectorCpp>& malicious_sample);
    double predict_threat_score_stub(const TrafficFeatureVectorCpp& traffic_features) const;


private:
    CoreInterface* core_recombinator_; // Non-owning
    std::list<FirewallRuleCpp> rules_; // Rules are kept sorted by priority (highest first)
    
    // For learning and adaptation
    std::deque<std::map<std::string, FirewallRuleCpp::PacketInfoValueVariant>> recent_traffic_log_packets_; // Store raw packet info
    std::deque<std::string> recent_traffic_log_actions_; // Store actions taken

    // Conceptual Threat Detection Model (e.g., simple statistical model or placeholder for ML model)
    std::map<std::string, double> threat_model_parameters_stub_; // e.g., {"avg_malicious_entropy": 0.8}
    double dynamic_threat_threshold_ = 0.7; // Threshold for classifying traffic as threat

    void process_simulated_traffic_packet_internal(); // Renamed
    void adapt_rules_based_on_activity_internal_stub(); // Renamed
    void sort_rules_by_priority();

    // Random number generation for traffic simulation
    mutable std::mt19937 rng_afw_;
    mutable std::uniform_real_distribution<double> uniform_01_afw_;
    mutable std::uniform_int_distribution<int> ip_octet_dist_afw_;
    mutable std::uniform_int_distribution<int> port_dist_afw_;
};
// eane_cpp_modules/adaptive_firewall_module/adaptive_firewall_module.cpp
#include "adaptive_firewall_module.h"
#include <algorithm> // For std::sort, std::find_if, std::remove_if
#include <chrono>    // For timestamps in rule creation
#include <iostream>  // For logging if core is null

AdaptiveFirewallModule::AdaptiveFirewallModule(CoreInterface* core)
    : core_recombinator_(core), 
      recent_traffic_log_packets_(1000), // Max 1000 packets in log
      recent_traffic_log_actions_(1000),
      rng_afw_(std::random_device{}()),
      uniform_01_afw_(0.0, 1.0),
      ip_octet_dist_afw_(1, 254),
      port_dist_afw_(1024, 65535) {
    
    // Initialize with a default "deny all" rule with the lowest priority
    FirewallRuleCpp default_deny;
    default_deny.id = "default_deny_all_cpp";
    default_deny.action = "DENY";
    default_deny.priority = -10000; // Very low priority
    default_deny.description = "Default deny all unmatched traffic (C++).";
    default_deny.enabled = true;
    default_deny.created_at_timestamp = core_recombinator_ ? core_recombinator_->get_current_timestamp() : 0.0;
    rules_.push_back(default_deny);
    // No need to sort yet as it's the only rule.

    if (core_recombinator_) {
        core_recombinator_->log_message("INFO", "AdaptiveFirewallCpp", "AdaptiveFirewallModule C++ initialized with default deny rule.");
    }
}

void AdaptiveFirewallModule::sort_rules_by_priority() {
    rules_.sort([](const FirewallRuleCpp& a, const FirewallRuleCpp& b) {
        return a.priority > b.priority; // Higher priority first
    });
}

void AdaptiveFirewallModule::update_logic() {
    if (!core_recombinator_) return;

    // 1. Process some simulated incoming traffic packets
    int packets_to_simulate = 5; // Simulate a few packets per update cycle
    for (int i = 0; i < packets_to_simulate; ++i) {
        process_simulated_traffic_packet_internal();
    }

    // 2. Periodically adapt rules or re-train threat model (conceptual)
    // if (core_recombinator_->get_current_cycle_num() % 100 == 0) { // Example frequency
    //    adapt_rules_based_on_activity_internal_stub();
    //    // if (recent_traffic_log_packets_.size() > 500) {
    //    //    // Conceptual: train_threat_model_stub using recent_traffic_log_packets_ and actions
    //    // }
    // }
}

void AdaptiveFirewallModule::process_simulated_traffic_packet_internal() {
    std::map<std::string, FirewallRuleCpp::PacketInfoValueVariant> packet;
    packet["timestamp_val"] = core_recombinator_->get_current_timestamp(); // Use _val for variant clarity

    // Generate diverse simulated packet data
    std::string src_ip_str = std::to_string(ip_octet_dist_afw_(rng_afw_)) + "." + 
                             std::to_string(ip_octet_dist_afw_(rng_afw_)) + "." + 
                             std::to_string(ip_octet_dist_afw_(rng_afw_)) + "." + 
                             std::to_string(ip_octet_dist_afw_(rng_afw_));
    packet["src_ip"] = src_ip_str;

    packet["dst_ip"] = (uniform_01_afw_(rng_afw_) < 0.7) ? "192.168.1.100" : // Common internal target
                       (std::string("10.0.0.") + std::to_string(ip_octet_dist_afw_(rng_afw_)));
    packet["src_port"] = port_dist_afw_(rng_afw_);
    packet["dst_port"] = (uniform_01_afw_(rng_afw_) < 0.6) ? 80 : // HTTP
                       ((uniform_01_afw_(rng_afw_) < 0.7) ? 443 : // HTTPS
                       port_dist_afw_(rng_afw_)); 
    packet["protocol"] = (uniform_01_afw_(rng_afw_) < 0.8) ? "TCP" : "UDP";
    packet["packet_size_val"] = static_cast<int>(uniform_01_afw_(rng_afw_) * 1400 + 40); // 40 to 1440 bytes
    
    std::string payload_content = "Simulated Data: ";
    if (uniform_01_afw_(rng_afw_) < 0.05) payload_content += "exec_payload_exploit"; // Suspicious
    else if (uniform_01_afw_(rng_afw_) < 0.15) payload_content += "user=admin;pass=123"; // Another suspicious
    else payload_content += "normal_web_traffic_content_example_" + std::to_string(static_cast<int>(uniform_01_afw_(rng_afw_)*1000));
    packet["payload_sample"] = payload_content;
    
    // Use the internal stub for threat score prediction
    TrafficFeatureVectorCpp tfv_for_score; // Populate this from 'packet'
    tfv_for_score.src_ip = src_ip_str;
    // ... (populate other tfv_for_score fields) ...
    tfv_for_score.payload_entropy = uniform_01_afw_(rng_afw_) * 8.0; // Simulated entropy
    packet["threat_score_val"] = predict_threat_score_stub(tfv_for_score);


    // Ensure rules are sorted (might be redundant if sort_rules_by_priority is called after every modification)
    // sort_rules_by_priority(); // Can be costly if called every packet

    std::string final_action_str = "default_pass_cpp"; // Fallback if no rule matches (conceptually, default deny is a rule)
    std::string matched_rule_id_str;

    for (FirewallRuleCpp& rule : rules_) { // Iterate by reference to modify hit_count
        if (rule.matches(packet)) {
            final_action_str = rule.action;
            matched_rule_id_str = rule.id;
            rule.hit_count++;
            rule.last_hit_timestamp = core_recombinator_->get_current_timestamp();
            break; // First matching rule determines action
        }
    }

    // Log traffic and action
    if (recent_traffic_log_packets_.size() >= recent_traffic_log_packets_.max_size()) {
        recent_traffic_log_packets_.pop_front();
        recent_traffic_log_actions_.pop_front();
    }
    recent_traffic_log_packets_.push_back(packet);
    recent_traffic_log_actions_.push_back(final_action_str + " (Rule: " + matched_rule_id_str + ")");

    if (final_action_str == "ALERT" || (final_action_str == "DENY" && matched_rule_id_str != "default_deny_all_cpp")) {
        EventDataCpp alert_event("firewall_incident_cpp"); // Renamed type
        alert_event.content["action_taken_str"] = final_action_str;
        alert_event.content["matched_rule_id_str"] = matched_rule_id_str;
        if(auto src_ip_opt = std::get_if<std::string>(&packet["src_ip"])) alert_event.content["src_ip_str_val"] = *src_ip_opt;
        // ... (add more packet details to event content if needed, carefully handling std::variant) ...
        alert_event.priority_label = (final_action_str == "ALERT") ? "high" : "medium";
        core_recombinator_->event_queue_put(alert_event);
        core_recombinator_->log_message("WARNING", "AdaptiveFirewallCpp", 
            "Traffic " + final_action_str + " by rule " + matched_rule_id_str + 
            ". SrcIP: " + (std::get_if<std::string>(&packet["src_ip"]) ? *std::get_if<std::string>(&packet["src_ip"]) : "N/A"));
    }
}

// --- Rule Management ---
std::string AdaptiveFirewallModule::add_rule(const FirewallRuleCpp& rule_data_in) {
    FirewallRuleCpp new_rule = rule_data_in; // Make a copy to modify
    
    // Ensure unique ID if not provided or if it collides
    auto it_find = std::find_if(rules_.begin(), rules_.end(), [&](const FirewallRuleCpp& r){ return r.id == new_rule.id; });
    if (new_rule.id.empty() || it_find != rules_.end()) {
        auto now_ms = std::chrono::duration_cast<std::chrono_milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count();
        new_rule.id = "fw_rule_cpp_gen_" + std::to_string(now_ms) + "_" + std::to_string(rules_.size());
    }

    new_rule.created_at_timestamp = core_recombinator_ ? core_recombinator_->get_current_timestamp() : 0.0;
    new_rule.compile_payload_regex(); // Compile regex string if provided

    rules_.push_back(new_rule);
    sort_rules_by_priority(); // Maintain order
    if (core_recombinator_) core_recombinator_->log_message("INFO", "AdaptiveFirewallCpp", "Added firewall rule: " + new_rule.id);
    return new_rule.id;
}

// ... (Implementations for remove_rule, update_rule, get_rule, get_all_rules - similar to Part 5,
//      ensuring sort_rules_by_priority() is called after modifications affecting order)

// --- Threat Model Stubs ---
// (Implementations for train_threat_model_stub, predict_threat_score_stub - similar to Part 5)
// eane_cpp_modules/timeseries_predictor_module/ts_types.h
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <map>
#include <any>      // For model_parameters_stub
#include <optional> // For last_prediction_stub to allow empty state

struct TimeSeriesDataCpp {
    std::string id;
    std::vector<double> timestamps; // Unix timestamps (double for precision)
    Eigen::VectorXd values;         // Time series values
    std::map<std::string, std::string> metadata; // e.g., {"unit": "celsius", "source": "SensorModule"}

    std::string model_type_id = "MovingAverage_Stub_Cpp"; // Default model
    std::any model_parameters_internal_stub; // Stores parameters specific to the model_type_id
                                         // e.g., for ARIMA: {p:int, d:int, q:int, coeffs:vector<double>}
                                         // e.g., for MovingAverage: {window_size:int}

    // Store last prediction as a map for flexibility:
    // {"forecast_times": Eigen::VectorXd, "forecast_values": Eigen::VectorXd, 
    //  "ci_lower_sim": Eigen::VectorXd, "ci_upper_sim": Eigen::VectorXd}
    std::optional<std::map<std::string, Eigen::VectorXd>> last_prediction_result_stub;
    int prediction_default_horizon = 5; // Default number of steps to predict

    size_t max_length = 200; // Max number of data points to keep in `values` and `timestamps`

    explicit TimeSeriesDataCpp(std::string series_id = "") : id(std::move(series_id)) {}
};
// eane_cpp_modules/timeseries_predictor_module/timeseries_predictor_module.h
#pragma once
#include "../core_interface.h"
#include "ts_types.h" // For TimeSeriesDataCpp
#include <map>
#include <string>
#include <random>   // For stubs and noise
#include <optional> // For return types

class TimeSeriesPredictorModule {
public:
    explicit TimeSeriesPredictorModule(CoreInterface* core);
    void update_logic();

    // --- API for Managing Time Series Data ---
    // Adds a new data point (timestamp, value) to an existing or new series.
    // If series_id doesn't exist, a new TimeSeriesDataCpp entry is created with default settings.
    bool add_or_update_data_point(const std::string& series_id, double timestamp, double value);
    
    // Registers or updates the full definition of a time series, including its model type and params.
    bool register_or_update_series_definition(const TimeSeriesDataCpp& ts_definition);

    // --- API for Model Training and Prediction (Stubs) ---
    // Conceptually trains a model for the specified series.
    bool train_model_for_series_stub(const std::string& series_id, 
                                     const std::string& model_type_to_use, // e.g., "ARIMA_Stub_Cpp"
                                     const std::map<std::string, std::any>& training_params = {});
    
    // Generates a forecast for the series. Returns the map from TimeSeriesDataCpp.last_prediction_result_stub.
    std::optional<std::map<std::string, Eigen::VectorXd>> predict_series_stub(
        const std::string& series_id, 
        std::optional<int> horizon_override = std::nullopt);

    // --- Accessor ---
    std::optional<TimeSeriesDataCpp> get_time_series_data_copy(const std::string& series_id) const; // Returns a copy
    std::vector<std::string> get_all_series_ids() const;

private:
    CoreInterface* core_recombinator_; // Non-owning
    std::map<std::string, TimeSeriesDataCpp> time_series_collection_;

    // Helper for stub predictions based on model_type
    std::map<std::string, Eigen::VectorXd> generate_stub_forecast(
        const TimeSeriesDataCpp& ts_data, int horizon) const;

    // Random number generation for stubs
    mutable std::mt19937 rng_tsp_;
    mutable std::normal_distribution<double> normal_dist_tsp_std_; // N(0,1)
};
// eane_cpp_modules/timeseries_predictor_module/timeseries_predictor_module.cpp
#include "timeseries_predictor_module.h"
#include <algorithm> // For std::sort, std::remove_if (not used yet but common)
#include <cmath>     // For std::sqrt, std::abs, std::round
#include <iostream>  // For logging if core is null

TimeSeriesPredictorModule::TimeSeriesPredictorModule(CoreInterface* core)
    : core_recombinator_(core), 
      rng_tsp_(std::random_device{}()), 
      normal_dist_tsp_std_(0.0, 1.0) {
    if (core_recombinator_) {
        core_recombinator_->log_message("INFO", "TimeSeriesPredictorCpp", "TimeSeriesPredictorModule C++ initialized.");
    }
}

void TimeSeriesPredictorModule::update_logic() {
    if (!core_recombinator_) return;

    // Periodically re-train or make predictions for key series (conceptual)
    // This is a STUB and would be driven by events or internal logic in a full system.
    // if (core_recombinator_->get_current_cycle_num() % 60 == 0) { // Example: every 60 core cycles
    //     for (auto const& [id, ts_data_const] : time_series_collection_) {
    //         if (ts_data_const.values.size() > 20 && ts_data_const.metadata.count("auto_predict_cpp")) {
    //             // Re-train (conceptually, if needed)
    //             // train_model_for_series_stub(id, ts_data_const.model_type_id);
                
    //             auto prediction_opt = predict_series_stub(id);
    //             if (prediction_opt && core_recombinator_) {
    //                 // Send event with new prediction
    //                 EventDataCpp pred_event("timeseries_prediction_update_cpp");
    //                 pred_event.content["series_id_str"] = id;
    //                 // Add actual prediction data to content (e.g., forecast_values as std::vector<double>)
    //                 // For simplicity, just sending a notification here
    //                 pred_event.content["forecast_horizon_val"] = static_cast<int>(prediction_opt->at("forecast_values").size());
    //                 core_recombinator_->event_queue_put(pred_event);
    //                 core_recombinator_->log_message("DEBUG", "TimeSeriesPredictorCpp", "Auto-predicted series: " + id);
    //             }
    //         }
    //     }
    // }
}

bool TimeSeriesPredictorModule::add_or_update_data_point(const std::string& series_id, double timestamp, double value) {
    if (series_id.empty()) {
        if (core_recombinator_) core_recombinator_->log_message("WARNING", "TimeSeriesPredictorCpp", "AddDataPoint: Series ID cannot be empty.");
        return false;
    }

    auto it = time_series_collection_.find(series_id);
    if (it == time_series_collection_.end()) {
        // Create a new series with default settings if it doesn't exist
        TimeSeriesDataCpp new_ts(series_id);
        new_ts.metadata["source_origin_cpp"] = "dynamic_add_data_point";
        time_series_collection_[series_id] = new_ts;
        it = time_series_collection_.find(series_id);
        if (core_recombinator_) core_recombinator_->log_message("INFO", "TimeSeriesPredictorCpp", "New time series '" + series_id + "' created implicitly by add_or_update_data_point.");
    }

    TimeSeriesDataCpp& ts_data = it->second; // Get reference to the series data

    // Add timestamp and value
    ts_data.timestamps.push_back(timestamp);
    
    Eigen::VectorXd old_values = ts_data.values;
    ts_data.values.resize(old_values.size() + 1);
    if (old_values.size() > 0) {
        ts_data.values.head(old_values.size()) = old_values;
    }
    ts_data.values(old_values.size()) = value;

    // Enforce max_length
    while (ts_data.timestamps.size() > ts_data.max_length && !ts_data.timestamps.empty()) {
        ts_data.timestamps.erase(ts_data.timestamps.begin());
        if (ts_data.values.size() > 0) { // Ensure values is not empty before tail
            ts_data.values = ts_data.values.tail(ts_data.values.size() - 1);
        }
    }
    // Invalidate previous prediction as new data is available
    ts_data.last_prediction_result_stub.reset();
    return true;
}

bool TimeSeriesPredictorModule::register_or_update_series_definition(const TimeSeriesDataCpp& ts_definition_in) {
    if (ts_definition_in.id.empty()) {
        if (core_recombinator_) core_recombinator_->log_message("WARNING", "TimeSeriesPredictorCpp", "RegisterSeries: Series ID in definition cannot be empty.");
        return false;
    }
    // This will overwrite if exists, or add if new.
    time_series_collection_[ts_definition_in.id] = ts_definition_in;
    if (core_recombinator_) core_recombinator_->log_message("INFO", "TimeSeriesPredictorCpp", "Registered/Updated time series definition for: " + ts_definition_in.id);
    return true;
}

bool TimeSeriesPredictorModule::train_model_for_series_stub(const std::string& series_id, 
                                                          const std::string& model_type_to_use,
                                                          const std::map<std::string, std::any>& training_params) {
    auto it = time_series_collection_.find(series_id);
    if (it == time_series_collection_.end()) {
        if (core_recombinator_) core_recombinator_->log_message("WARNING", "TimeSeriesPredictorCpp", "TrainModel: Series ID '" + series_id + "' not found.");
        return false;
    }
    TimeSeriesDataCpp& ts_data = it->second;

    if (ts_data.values.size() < 10) { // Arbitrary minimum for stub training
        if (core_recombinator_) core_recombinator_->log_message("INFO", "TimeSeriesPredictorCpp", "TrainModel: Not enough data for series '" + series_id + "' (need at least 10 points).");
        return false;
    }

    ts_data.model_type_id = model_type_to_use;
    ts_data.model_parameters_internal_stub.reset(); // Clear old params

    if (core_recombinator_) core_recombinator_->log_message("INFO", "TimeSeriesPredictorCpp", "STUB: Training model '" + model_type_to_use + "' for series '" + series_id + "'.");

    if (model_type_to_use == "ARIMA_Stub_Cpp") {
        std::map<std::string, std::any> arima_params_sim;
        arima_params_sim["p_val"] = static_cast<int>(rng_tsp_() % 3 + 1); // p in [1,3]
        arima_params_sim["d_val"] = static_cast<int>(rng_tsp_() % 2);     // d in [0,1]
        arima_params_sim["q_val"] = static_cast<int>(rng_tsp_() % 3);     // q in [0,2]
        // Simulate some coefficients
        std::vector<double> ar_coeffs(std::any_cast<int>(arima_params_sim["p_val"]));
        for(double& c : ar_coeffs) c = uniform_01_afw_(rng_tsp_) * 0.6 - 0.1; // [-0.1, 0.5]
        arima_params_sim["ar_coeffs_vec_sim"] = ar_coeffs;
        // ... (similar for ma_coeffs if q > 0) ...
        arima_params_sim["variance_sim_val"] = std::pow(uniform_01_afw_(rng_tsp_) * 0.2 + 0.01, 2.0);
        ts_data.model_parameters_internal_stub = arima_params_sim;
    } else if (model_type_to_use == "MovingAverage_Stub_Cpp") {
        int window = 5; // Default window
        if (training_params.count("window_size_val")) {
            try { window = std::any_cast<int>(training_params.at("window_size_val")); }
            catch (const std::bad_any_cast&) { /* use default */ }
        }
        ts_data.model_parameters_internal_stub = std::max(1, window);
    } else {
        if (core_recombinator_) core_recombinator_->log_message("WARNING", "TimeSeriesPredictorCpp", "TrainModel: Model type '" + model_type_to_use + "' not supported by stub.");
        return false;
    }
    ts_data.last_prediction_result_stub.reset(); // Training invalidates last prediction
    return true;
}


std::optional<std::map<std::string, Eigen::VectorXd>> TimeSeriesPredictorModule::predict_series_stub(
    const std::string& series_id, std::optional<int> horizon_override) {
    
    auto it = time_series_collection_.find(series_id);
    if (it == time_series_collection_.end()) {
        if (core_recombinator_) core_recombinator_->log_message("WARNING", "TimeSeriesPredictorCpp", "Predict: Series ID '" + series_id + "' not found.");
        return std::nullopt;
    }
    TimeSeriesDataCpp& ts_data = it->second; // Modifiable to store prediction

    if (ts_data.values.size() < 2) { // Need at least a couple of points for any sensible stub prediction
        if (core_recombinator_) core_recombinator_->log_message("INFO", "TimeSeriesPredictorCpp", "Predict: Not enough data for series '" + series_id + "'.");
        return std::nullopt;
    }

    int horizon = horizon_override.value_or(ts_data.prediction_default_horizon);
    if (horizon <= 0) horizon = ts_data.prediction_default_horizon;

    ts_data.last_prediction_result_stub = generate_stub_forecast(ts_data, horizon);
    return ts_data.last_prediction_result_stub;
}

std::map<std::string, Eigen::VectorXd> TimeSeriesPredictorModule::generate_stub_forecast(
    const TimeSeriesDataCpp& ts_data, int horizon) const {
    
    std::map<std::string, Eigen::VectorXd> forecast_result;
    Eigen::VectorXd forecast_values(horizon);
    Eigen::VectorXd forecast_times(horizon);
    Eigen::VectorXd ci_lower_sim(horizon); // Conceptual confidence intervals
    Eigen::VectorXd ci_upper_sim(horizon);

    double last_value = ts_data.values.size() > 0 ? ts_data.values(ts_data.values.size() - 1) : 0.0;
    double last_timestamp = ts_data.timestamps.empty() ? 0.0 : ts_data.timestamps.back();
    double avg_dt = 1.0; // Default time step
    if (ts_data.timestamps.size() >= 2) {
        avg_dt = (ts_data.timestamps.back() - ts_data.timestamps.front()) / static_cast<double>(ts_data.timestamps.size() - 1);
        if (avg_dt <= 0) avg_dt = 1.0; // Prevent non-positive dt
    }

    if (ts_data.model_type_id == "ARIMA_Stub_Cpp") {
        // Simulate ARIMA: tends to revert to mean or project trend with noise
        double trend_sim = 0.0;
        if (ts_data.values.size() >= 5) { // Estimate simple trend from last 5 points
            trend_sim = (ts_data.values(ts_data.values.size()-1) - ts_data.values(ts_data.values.size()-5)) / 4.0;
        }
        for (int h = 0; h < horizon; ++h) {
            forecast_values(h) = last_value + trend_sim * (h + 1) + normal_dist_tsp_std_(rng_tsp_) * 0.1; // Add noise
        }
    } else if (ts_data.model_type_id == "MovingAverage_Stub_Cpp") {
        int window = 5;
        if (ts_data.model_parameters_internal_stub.has_value()) {
            try { window = std::any_cast<int>(ts_data.model_parameters_internal_stub); } catch(const std::bad_any_cast&) {}
        }
        window = std::max(1, std::min(window, static_cast<int>(ts_data.values.size())));
        
        Eigen::VectorXd recent_values = ts_data.values.tail(window);
        double moving_avg = recent_values.mean();
        for (int h = 0; h < horizon; ++h) {
            // For multi-step, MA would use its own previous forecasts. This is simplified.
            forecast_values(h) = moving_avg + normal_dist_tsp_std_(rng_tsp_) * 0.05;
        }
    } else { // Default: Naive forecast (last value) with noise
        for (int h = 0; h < horizon; ++h) {
            forecast_values(h) = last_value + normal_dist_tsp_std_(rng_tsp_) * 0.2; // Larger noise for naive
        }
    }

    for (int h = 0; h < horizon; ++h) {
        forecast_times(h) = last_timestamp + avg_dt * (h + 1);
        // Simulated confidence interval (e.g., +/- 1.96 * simulated_std_dev)
        double simulated_std_dev = 0.1 + 0.05 * std::sqrt(static_cast<double>(h+1)); // Wider CI for longer horizon
        ci_lower_sim(h) = forecast_values(h) - 1.96 * simulated_std_dev;
        ci_upper_sim(h) = forecast_values(h) + 1.96 * simulated_std_dev;
    }

    forecast_result["forecast_values"] = forecast_values;
    forecast_result["forecast_times"] = forecast_times;
    forecast_result["ci_lower_sim"] = ci_lower_sim;
    forecast_result["ci_upper_sim"] = ci_upper_sim;
    return forecast_result;
}

// --- Accessor ---
std::optional<TimeSeriesDataCpp> TimeSeriesPredictorModule::get_time_series_data_copy(const std::string& series_id) const {
    auto it = time_series_collection_.find(series_id);
    if (it != time_series_collection_.end()) {
        return it->second; // Returns a copy
    }
    return std::nullopt;
}

std::vector<std::string> TimeSeriesPredictorModule::get_all_series_ids() const {
    std::vector<std::string> ids;
    ids.reserve(time_series_collection_.size());
    for(auto const& [id, val] : time_series_collection_){
        ids.push_back(id);
    }
    return ids;
}
// eane_cpp_modules/controlled_mutation_generator/mugen_types.h
#pragma once
#include <string>
#include <vector>
#include <map>
#include <any>      // For original_value, mutated_value if truly generic
#include <variant>  // C++17, preferred for known set of parameter types
#include <chrono>   // For unique ID generation

// Define a variant for parameter values MuGen can handle.
// Add more types as needed (e.g., Eigen::VectorXd for array parameters).
using MuGenParameterValueCpp = std::variant<
    std::monostate, // Represents an uninitialized or null-like state
    bool,
    int,
    long long, // For larger integer parameters
    double,
    std::string,
    std::vector<double> // For list-like numerical parameters
    // If Eigen::VectorXd is needed: #include <Eigen/Dense> and add it here.
>;

struct MutationCandidateCpp {
    std::string candidate_id;
    std::string target_type_str; // e.g., "system_parameter_gs", "module_parameter_runtime", "architectural_link_conceptual"
    std::string target_identifier_str; // e.g., "GlobalSelfStateCpp.coherence_score", "LearningModuleCpp.alpha_lr_fwe_"
    std::string parameter_name_str;    // Specific sub-parameter if target_identifier is a complex object/module

    MuGenParameterValueCpp original_value_variant;
    MuG```cpp
// eane_cpp_modules/controlled_mutation_generator/mugen_types.h
#pragma once
#include <string>
#include <vector>
#include <map>
#include <any>      // Fallback for truly generic values, but variant is preferred
#include <variant>  // C++17
#include <chrono>   // For unique ID generation
#include <optional> // For optional fields

// Define a variant for parameter values MuGen can handle.
// Add more types as needed (e.g., Eigen::VectorXd for array parameters).
using MuGenParameterValueCpp = std::variant<
    std::monostate, // Represents an uninitialized or null-like state
    bool,
    int,
    long long, // For larger integer parameters
    double,
    std::string,
    std::vector<double> // For list-like numerical parameters
    // If Eigen::VectorXd is needed: #include <Eigen/Dense> and add it here.
    // std::map<std::string, double> // For dict-like parameters
>;

struct MutationCandidateCpp {
    std::string candidate_id;
    std::string target_type_str; // e.g., "system_parameter_gs", "module_parameter_runtime", "architectural_link_conceptual"
    std::string target_identifier_str; // e.g., "GlobalSelfStateCpp.coherence_score", "LearningModuleCpp.alpha_lr_fwe_"
    std::optional<std::string> parameter_name_opt_str; // Specific sub-parameter if target_identifier is a complex object/module

    MuGenParameterValueCpp original_value_variant;
    MuGenParameterValueCpp mutated_value_variant;

    std::string mutation_operator_used_str; // e.g., "gaussian_perturbation_cpp", "value_swap_cpp"

    // For MuGen V2.0 (Python) compatibility - simulated values
    std::map<std::string, double> predicted_impact_vector_sim_map; // {"dim1_impact_sim": val1, ...}
    double overall_predicted_desirability_sim_val = 0.0;
    double simulation_confidence_sim_val = 0.0;
    bool meets_improvement_threshold_sim_flag = false;

    double timestamp_val = 0.0; // Unix timestamp
    std::optional<std::string> context_hash_at_proposal_sim_str;
    std::vector<std::string> tags_vec;

    MutationCandidateCpp() {
        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count();
        candidate_id = "mugen_cand_cpp_" + std::to_string(now_ms) + "_" + std::to_string(rand() % 1000);
    }
};
// eane_cpp_modules/controlled_mutation_generator/controlled_mutation_generator.h
#pragma once
#include "../core_interface.h" // For CoreInterface, GlobalSelfStateCpp
#include "mugen_types.h"       // For MutationCandidateCpp
#include <vector>
#include <string>
#include <map>
#include <random>
#include <Eigen/Dense> // For abstract_genome (if it's an Eigen vector)
#include <any>         // For sem_context and surrogate_models_stub_

class ControlledMutationGenerator {
public:
    ControlledMutationGenerator(CoreInterface* core, 
                                int abstract_genome_dim_hint = 50); // Hint for genome translation

    void update_logic(); // Can be used for periodic model refinement or proactive generation

    // --- Main API for SelfEvolutionModule (SEM) ---
    // Translates an abstract genome into a concrete mutation candidate, evaluates it,
    // and if desirable, proposes it (e.g., by sending an event).
    // sem_context: map from SEM containing fitness, landscape_id, current_gs_snapshot (as std::any)
    // Returns <proposed_mutation_id_or_empty_if_not_proposed, predicted_desirability_score>
    std::pair<std::string, double> generate_and_propose_mutation_from_abstract_genome(
        const Eigen::VectorXd& abstract_genome, // Genome from SEM's IndividualCpp
        const std::map<std::string, std::any>& sem_context_from_caller);

    // --- API for evaluating externally defined mutations (e.g., from EPN or human input) ---
    // Takes a template for a mutation, fills in predictions, and returns the evaluated candidate.
    MutationCandidateCpp evaluate_specific_mutation_candidate_with_surrogate(
        const MutationCandidateCpp& concrete_mutation_template, // Input candidate (may lack predictions)
        const GlobalSelfStateCpp& current_gs_for_context);    // GS context for evaluation

private:
    CoreInterface* core_recombinator_; // Non-owning
    int abstract_genome_dim_hint_;

    // Surrogate Models (Conceptual Stubs)
    // Key: String identifier for the target of prediction (e.g., "GlobalSelfStateCpp.coherence_score_impact")
    // Value: std::any holding the "model" (e.g., simple weights, a small ML model object if using a C++ ML lib)
    std::map<std::string, std::any> surrogate_models_internal_stub_;
    std::map<std::string, double> surrogate_model_confidence_internal_stub_; // Confidence in each surrogate

    double desirability_threshold_for_proposal_ = 0.6; // Min desirability to propose
    // int max_mutation_proposals_per_cycle_ = 3; // If MuGen was proactive

    // --- Internal Helper Methods ---
    MutationCandidateCpp translate_abstract_genome_to_concrete_mutation_internal_stub(
        const Eigen::VectorXd& abstract_genome,
        const std::string& fitness_landscape_id_from_sem_sim,
        const GlobalSelfStateCpp& gs_context_for_translation_sim) const;

    std::map<std::string, double> predict_impact_with_surrogates_internal_stub(
        const MutationCandidateCpp& candidate_to_evaluate,
        const GlobalSelfStateCpp& gs_context_for_prediction_sim) const;

    double calculate_overall_desirability_internal_stub(
        const std::map<std::string, double>& predicted_impact_vector,
        const std::string& fitness_landscape_id_from_sem_sim) const; // To get objective weights

    // Random number generation for stubs
    mutable std::mt19937 rng_mugen_internal_;
    mutable std::uniform_real_distribution<double> uniform_dist_mugen_01_;
    mutable std::normal_distribution<double> normal_dist_mugen_std_;
};
// eane_cpp_modules/controlled_mutation_generator/controlled_mutation_generator.cpp
#include "controlled_mutation_generator.h"
#include <cmath>     // For std::tanh, std::abs, std::clamp
#include <numeric>   // For std::accumulate (if needed for desirability)
#include <algorithm> // For std::transform (if needed)
#include <iostream>  // For logging if core_ is null

// Helper to convert MuGenParameterValueCpp to string (for logging/debug)
std::string mugen_param_value_to_string(const MuGenParameterValueCpp& val) {
    return std::visit([](auto&& arg) -> std::string {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::monostate>) return "<monostate>";
        else if constexpr (std::is_same_v<T, bool>) return arg ? "true" : "false";
        else if constexpr (std::is_same_v<T, std::vector<double>>) {
            std::string s = "[";
            for(size_t i=0; i<arg.size(); ++i) s += std::to_string(arg[i]) + (i==arg.size()-1 ? "" : ", ");
            s += "]";
            return s;
        }
        // Check for int, long long, double, string before a generic to_string
        else if constexpr (std::is_arithmetic_v<T> || std::is_same_v<T, std::string>) {
             if constexpr (std::is_same_v<T, std::string>) return "\"" + arg + "\""; // Quote strings
             else return std::to_string(arg);
        }
        else return "<unhandled_variant_type>";
    }, val);
}


ControlledMutationGenerator::ControlledMutationGenerator(CoreInterface* core, int abstract_genome_dim_hint)
    : core_recombinator_(core), abstract_genome_dim_hint_(abstract_genome_dim_hint),
      rng_mugen_internal_(std::random_device{}()), 
      uniform_dist_mugen_01_(0.0, 1.0),
      normal_dist_mugen_std_(0.0, 1.0) {

    // Initialize conceptual surrogate models (e.g., simple linear weights for impact prediction)
    if (abstract_genome_dim_hint_ > 0) {
        surrogate_models_internal_stub_["gs_coherence_impact_sim_model"] = Eigen::VectorXd::Random(abstract_genome_dim_hint_) * 0.05;
        surrogate_model_confidence_internal_stub_["gs_coherence_impact_sim_model"] = 0.65;

        surrogate_models_internal_stub_["gs_entropy_impact_sim_model"] = Eigen::VectorXd::Random(abstract_genome_dim_hint_) * 0.03;
        surrogate_model_confidence_internal_stub_["gs_entropy_impact_sim_model"] = 0.60;
        
        surrogate_models_internal_stub_["gs_phi_functional_impact_sim_model"] = Eigen::VectorXd::Random(abstract_genome_dim_hint_) * 0.08;
        surrogate_model_confidence_internal_stub_["gs_phi_functional_impact_sim_model"] = 0.70;
    }

    if (core_recombinator_) {
        core_recombinator_->log_message("INFO", "MuGenCpp", 
            "ControlledMutationGenerator C++ (MuGen V2.0 Stub) initialized. Genome Hint Dim: " + std::to_string(abstract_genome_dim_hint_));
    }
}

void ControlledMutationGenerator::update_logic() {
    if (!core_recombinator_) return;
    // MuGen is primarily reactive. This logic could be used for:
    // 1. Listening to "mugen_retrain_surrogate_model_request_cpp" events from LearningModule.
    // 2. Proactively generating "exploratory" mutations if system is highly stable or stagnated.
    //    (This would require coordination with SEM or a stagnation detection mechanism).

    // Example: Conceptual check for retraining request (event would come from Python LM)
    // EventDataCpp retrain_event;
    // if (core_recombinator_->event_queue_get_specific_cpp("mugen_retrain_surrogate_request_cpp", retrain_event, 0.001)) {
    //     std::string model_to_retrain_id_str = "unknown_model";
    //     if(retrain_event.content.count("surrogate_model_id_str")) {
    //         try { model_to_retrain_id_str = std::get<std::string>(retrain_event.content.at("surrogate_model_id_str"));} catch(const std::bad_variant_access&){}
    //     }
    //     // ... get new training data (e.g., from DKPM via an ID in the event) ...
    //     // ... retrain the specific surrogate_models_internal_stub_[model_to_retrain_id_str] ...
    //     core_recombinator_->log_message("INFO", "MuGenCpp", "STUB: Received request to retrain surrogate model: " + model_to_retrain_id_str);
    // }
}

std::pair<std::string, double> ControlledMutationGenerator::generate_and_propose_mutation_from_abstract_genome(
    const Eigen::VectorXd& abstract_genome,
    const std::map<std::string, std::any>& sem_context_from_caller) {

    if (!core_recombinator_) return {"", 0.0};
    if (abstract_genome.size() == 0 || (abstract_genome_dim_hint_ > 0 && abstract_genome.size() != abstract_genome_dim_hint_)) {
        core_recombinator_->log_message("ERROR", "MuGenCpp", "GenerateFromGenome: Abstract genome dimension mismatch or empty. Expected: " + std::to_string(abstract_genome_dim_hint_) + ", Got: " + std::to_string(abstract_genome.size()));
        return {"", 0.0};
    }

    std::string fitness_landscape_id = "default_landscape_for_mugen_cpp";
    if (sem_context_from_caller.count("current_fitness_landscape_id_sem_str")) { // Assuming SEM passes ID as string
        try { fitness_landscape_id = std::any_cast<std::string>(sem_context_from_caller.at("current_fitness_landscape_id_sem_str")); }
        catch(const std::bad_any_cast&) { /* use default */ }
    }
    
    // Use current GlobalSelfState for context in translation and impact prediction
    const GlobalSelfStateCpp& current_gs_snapshot = core_recombinator_->get_global_state_const();

    MutationCandidateCpp concrete_candidate = translate_abstract_genome_to_concrete_mutation_internal_stub(
                                                  abstract_genome, fitness_landscape_id, current_gs_snapshot);
    concrete_candidate.timestamp_val = core_recombinator_->get_current_timestamp();
    // conceptual_hash = hash_current_gs(current_gs_snapshot);
    // concrete_candidate.context_hash_at_proposal_sim_str = conceptual_hash;

    concrete_candidate.predicted_impact_vector_sim_map = predict_impact_with_surrogates_internal_stub(
                                                            concrete_candidate, current_gs_snapshot);
    concrete_candidate.overall_predicted_desirability_sim_val = calculate_overall_desirability_internal_stub(
                                                                    concrete_candidate.predicted_impact_vector_sim_map, fitness_landscape_id);
    
    // Aggregate confidence from surrogate models used
    double total_confidence_sum = 0.0;
    int num_confidences = 0;
    for(const auto& impact_pair : concrete_candidate.predicted_impact_vector_sim_map){
        // Conceptual: find surrogate model for this impact dimension and get its confidence
        // Example: if impact_pair.first is "coherence_score_delta_sim", find confidence for "gs_coherence_impact_sim_model"
        // This mapping needs to be more robust.
        if (surrogate_model_confidence_internal_stub_.count("gs_coherence_impact_sim_model") && impact_pair.first.find("coherence") != std::string::npos) { // very loose match
            total_confidence_sum += surrogate_model_confidence_internal_stub_.at("gs_coherence_impact_sim_model");
            num_confidences++;
        } // ... add for other dimensions ...
    }
    concrete_candidate.simulation_confidence_sim_val = (num_confidences > 0) ? (total_confidence_sum / num_confidences) : 0.5;

    concrete_candidate.meets_improvement_threshold_sim_flag = 
        concrete_candidate.overall_predicted_desirability_sim_val >= desirability_threshold_for_proposal_;

    if (core_recombinator_) {
        core_recombinator_->log_message("INFO", "MuGenCpp", 
            "Generated Mutation Candidate from Genome: ID=" + concrete_candidate.candidate_id +
            ", Desirability=" + std::to_string(concrete_candidate.overall_predicted_desirability_sim_val) +
            ", Confidence=" + std::to_string(concrete_candidate.simulation_confidence_sim_val) +
            ", MeetsThreshold=" + (concrete_candidate.meets_improvement_threshold_sim_flag ? "Yes" : "No"));
    }

    if (concrete_candidate.meets_improvement_threshold_sim_flag) {
        EventDataCpp proposal_event("mugen_mutation_proposal_for_epn_py_counterpart"); // To Python EPN
        proposal_event.priority_label = "medium";
        proposal_event.source_module = "MuGenCpp";
        
        // For Python, we need to convert MutationCandidateCpp to a Python-friendly format (e.g., dict)
        // This is handled by Pybind11 wrapper for the return type.
        // Here, we just send key info in the event content.
        proposal_event.content["candidate_id_str"] = concrete_candidate.candidate_id;
        proposal_event.content["desirability_score_val"] = concrete_candidate.overall_predicted_desirability_sim_val;
        proposal_event.content["target_type_str_val"] = concrete_candidate.target_type_str; // Example
        proposal_event.content["target_identifier_str_val"] = concrete_candidate.target_identifier_str; // Example
        // The full MutationCandidateCpp object would be returned to SEM (Python) if SEM calls this C++ function directly via Pybind.
        // If MuGen C++ is proactive, it needs a way to store the full candidate for EPN Python to fetch.

        core_recombinator_->event_queue_put(proposal_event);
        return {concrete_candidate.candidate_id, concrete_candidate.overall_predicted_desirability_sim_val};
    }
    return {"", concrete_candidate.overall_predicted_desirability_sim_val}; // Not proposed
}

// ... (Implementations for evaluate_specific_mutation_candidate, translate_genome, predict_impact, calculate_desirability
//      will be stubs similar to those in Part 7, but using the C++ types.)

// (Rest of MuGen C++ implementation stubs would go here)
// eane_cpp_modules/lyuk_parser/lyuk_ast_types.h
#pragma once
#include <string>
#include <vector>
#include <map>
#include <variant>  // C++17
#include <memory>   // For std::shared_ptr
#include <optional> // C++17

// Forward declaration for recursive structures (e.g., CodeBlock containing other nodes)
struct LyukBasePrimitiveASTNodeCpp;

// Using a variant for Lyuk data values
using LyukValueVariantCpp = std::variant<
    std::monostate, // Represents nil or an uninitialized Lyuk value
    bool,           // Lyuk Boolean
    long long,      // Lyuk Number (integer part)
    double,         // Lyuk Number (floating-point part)
    std::string,    // Lyuk String, Symbol, Emotion (as string label), Concept (as string label)
    std::vector<std::shared_ptr<LyukBasePrimitiveASTNodeCpp>>, // Lyuk CodeBlock (sequence of AST nodes)
    std::vector<std::variant<std::shared_ptr<LyukBasePrimitiveASTNodeCpp>, struct LyukDataTypeCpp>> // Lyuk List (mixed content)
    // FrameHandle could be a string ID or a dedicated struct pointer
>;

struct LyukDataTypeCpp {
    std::string type_name_str; // "Number", "String", "Symbol", "Emotion", "Concept", "CodeBlock", "List", "Boolean", "FrameHandle"
    LyukValueVariantCpp value_variant;
    std::map<std::string, std::string> metadata_map; // Simple string key-value metadata
    bool is_literal_flag = true;
    double inferred_certainty_val = 1.0;
};

struct LyukBasePrimitiveASTNodeCpp {
    std::string primitive_name_str; // e.g., "FELT", "KNOW", "FRAME", "LET", "IF"
    std::string raw_arguments_original_text_str; 
    
    // Parsed arguments can be other AST nodes (for expressions/blocks) or resolved LyukDataTypes
    std::vector<std::variant<std::shared_ptr<LyukBasePrimitiveASTNodeCpp>, LyukDataTypeCpp>> parsed_arguments_vec;
    
    std::optional<std::string> destination_variable_name_opt_str; // For LET, TRANSFORM
    std::map<std::string, std::string> metadata_tags_map; // Tags for this specific primitive invocation
    std::vector<std::string> semantic_errors_found_vec;
    int source_line_ref_num = -1; // Line number in the original Lyuk code
    std::string unique_node_id_str;

    LyukBasePrimitiveASTNodeCpp(std::string name = "") : primitive_name_str(std::move(name)) {
        static long long ast_id_counter_cpp = 0;
        unique_node_id_str = "ast_cpp_node_" + std::to_string(ast_id_counter_cpp++);
    }
    virtual ~LyukBasePrimitiveASTNodeCpp() = default;
    
    // For simple RTTI/identification if needed, or for a virtual 'execute' method
    virtual std::string get_ast_node_type_str() const { return "BasePrimitiveASTNodeCpp"; }
};

// --- Specific AST Node Types (Examples - expand as needed) ---
struct LyukFrameASTNodeCpp : public LyukBasePrimitiveASTNodeCpp {
    std::optional<std::variant<std::shared_ptr<LyukBasePrimitiveASTNodeCpp>, LyukDataTypeCpp>> frame_name_arg_opt_variant; // Symbol or String
    std::vector<std::shared_ptr<LyukBasePrimitiveASTNodeCpp>> body_primitives_vec;
    // std::map<std::string, std::any> context_activation_params_map_sim; // Conceptual
    // std::shared_ptr<LyukBasePrimitiveASTNodeCpp> entry_conditions_ast_ptr_sim;
    // std::shared_ptr<LyukBasePrimitiveASTNodeCpp> exit_conditions_ast_ptr_sim;

    LyukFrameASTNodeCpp() : LyukBasePrimitiveASTNodeCpp("FRAME") {}
    std::string get_ast_node_type_str() const override { return "FrameASTNodeCpp"; }
};

struct LyukLetASTNodeCpp : public LyukBasePrimitiveASTNodeCpp {
    // destination_variable_name_opt_str from base class is used for the variable name
    std::optional<std::variant<std::shared_ptr<LyukBasePrimitiveASTNodeCpp>, LyukDataTypeCpp>> value_to_assign_arg_opt_variant;
    std::string scope_lifetime_str = "current_frame_cpp"; // "current_frame_cpp", "module_global_cpp", etc.

    LyukLetASTNodeCpp() : LyukBasePrimitiveASTNodeCpp("LET") {}
     std::string get_ast_node_type_str() const override { return "LetASTNodeCpp"; }
};

// Add LyukIfASTNodeCpp, LyukLoopASTNodeCpp, etc. as per the Python definitions.
// eane_cpp_modules/lyuk_parser/lyuk_parser.h
#pragma once
#include "../core_interface.h"
#include "lyuk_ast_types.h" // For AST node structures
#include <string>
#include <vector>
#include <memory> // For std::shared_ptr
#include <map>    // For interpretation results
#include <any>    // For interpretation results if very generic

// This class will be a STUB for both LMI (Lyuk Multilevel Interpreter)
// and LTC (Lyuk Transcompiler) functionalities.
class LyukParser {
public:
    explicit LyukParser(CoreInterface* core);

    // --- Parsing (LMI Phase 1 - Syntactic Parsing) ---
    // Parses a string of Lyuk code into a sequence of AST nodes.
    // Returns an empty vector on major parsing failure. Errors are logged.
    std::vector<std::shared_ptr<LyukBasePrimitiveASTNodeCpp>> parse_lyuk_code_to_ast_stub(
        const std::string& lyuk_code_str);

    // --- Semantic Interpretation (LMI Phase 2 - Conceptual) ---
    // Takes an AST sequence and performs semantic analysis, type checking, scope resolution (stub).
    // Returns a map summarizing the interpretation (e.g., identified entities, intentions).
    // The std::any here is a placeholder for more structured interpretation results.
    std::map<std::string, std::any> interpret_ast_semantically_stub(
        const std::vector<std::shared_ptr<LyukBasePrimitiveASTNodeCpp>>& ast_sequence);

    // --- Transcompilation (LTC - Conceptual) ---
    // Converts an AST sequence into a human-readable string representation (e.g., pseudocode).
    std::string transcompile_ast_to_human_readable_stub(
        const std::vector<std::shared_ptr<LyukBasePrimitiveASTNodeCpp>>& ast_sequence,
        const std::string& target_representation_type = "pseudocode_cpp_detail_stub");

    // (Future) LTC V2.5: Generate structured data for visualization
    // std::map<std::string, std::any> generate_visualization_data_from_ast_stub(
    //    const std::vector<std::shared_ptr<LyukBasePrimitiveASTNodeCpp>>& ast_sequence_vis);

private:
    CoreInterface* core_recombinator_; // Non-owning

    // --- Internal Parsing Helpers (Stubs) ---
    // In a real parser, these would involve a lexer and parser (e.g., using ANTLR, Flex/Bison, or hand-rolled).
    std::shared_ptr<LyukBasePrimitiveASTNodeCpp> parse_single_lyuk_primitive_line_stub(const std::string& lyuk_line_str, int line_num);
    std::string extract_primitive_name_from_line_stub(const std::string& lyuk_line_str) const;
    std::string extract_raw_arguments_from_line_stub(const std::string& lyuk_line_str) const;
    // More helpers for parsing arguments into LyukDataTypeCpp or nested ASTs.
    std::vector<std::variant<std::shared_ptr<LyukBasePrimitiveASTNodeCpp>, LyukDataTypeCpp>> parse_arguments_stub(const std::string& raw_args_str);
};
// eane_cpp_modules/controlled_mutation_generator/controlled_mutation_generator.cpp
// (Continuation from Part 10, previous response)
#include "controlled_mutation_generator.h" // Already included

MutationCandidateCpp ControlledMutationGenerator::translate_abstract_genome_to_concrete_mutation_internal_stub(
    const Eigen::VectorXd& abstract_genome,
    const std::string& fitness_landscape_id_from_sem_sim,
    const GlobalSelfStateCpp& gs_context_for_translation_sim) const {

    MutationCandidateCpp candidate; // ID is auto-generated
    candidate.tags_vec.push_back("from_abstract_genome_cpp_stub");
    candidate.timestamp_val = core_recombinator_ ? core_recombinator_->get_current_timestamp() : 0.0;

    if (abstract_genome.size() == 0 || (abstract_genome_dim_hint_ > 0 && abstract_genome.size() != abstract_genome_dim_hint_)) {
        if(core_recombinator_) core_recombinator_->log_message("ERROR", "MuGenCpp", "TranslateGenome: Genome size mismatch in stub.");
        candidate.target_type_str = "translation_error_cpp";
        return candidate;
    }

    // STUB: Very simplified translation logic.
    // A real system would have complex rules or a learned model to map genome to concrete mutations.
    int gene_idx = 0;
    auto get_gene_val = [&](bool increment = true) { // Helper to get gene and advance index safely
        double val = 0.5; // Default
        if(abstract_genome.size() > 0) val = abstract_genome(gene_idx % abstract_genome.size());
        if(increment) gene_idx++;
        return val;
    };

    double target_type_gene = get_gene_val();
    double target_param_gene = get_gene_val();
    double operator_gene = get_gene_val();
    double change_magnitude_gene = get_gene_val(); // In [0,1]

    if (target_type_gene < 0.4) {
        candidate.target_type_str = "system_parameter_gs_cpp";
        std::vector<std::string> gs_params = {"GlobalSelfStateCpp.coherence_score", "GlobalSelfStateCpp.system_entropy", "GlobalSelfStateCpp.motivacion", "GlobalSelfStateCpp.phi_funcional_score"};
        candidate.target_identifier_str = gs_params[static_cast<int>(target_param_gene * gs_params.size()) % gs_params.size()];
        
        // Get original value (conceptual)
        if (candidate.target_identifier_str == "GlobalSelfStateCpp.coherence_score") candidate.original_value_variant = gs_context_for_translation_sim.coherence_score;
        else if (candidate.target_identifier_str == "GlobalSelfStateCpp.system_entropy") candidate.original_value_variant = gs_context_for_translation_sim.system_entropy;
        else if (candidate.target_identifier_str == "GlobalSelfStateCpp.motivacion") candidate.original_value_variant = gs_context_for_translation_sim.motivacion;
        else if (candidate.target_identifier_str == "GlobalSelfStateCpp.phi_funcional_score") candidate.original_value_variant = gs_context_for_translation_sim.phi_funcional_score;
        else candidate.original_value_variant = 0.5; // Default original

        double current_val = std::holds_alternative<double>(candidate.original_value_variant) ? std::get<double>(candidate.original_value_variant) : 0.5;
        double change = (change_magnitude_gene - 0.5) * 0.2; // Max change +/- 0.1
        candidate.mutated_value_variant = std::clamp(current_val + change, 0.0, 1.0);

    } else if (target_type_gene < 0.8) {
        candidate.target_type_str = "module_parameter_runtime_cpp";
        // Example module parameters (these would need to be actual accessible parameters)
        std::vector<std::string> mod_params = {"LearningModuleCpp.alpha_lr_fwe_stub", "SelfEvolutionModuleCpp.mutation_rate_base_stub", "EmotionRegulationModuleCpp.kp_erm_stub"};
        candidate.target_identifier_str = mod_params[static_cast<int>(target_param_gene * mod_params.size()) % mod_params.size()];
        candidate.original_value_variant = 0.1; // Placeholder original
        candidate.mutated_value_variant = std::clamp(0.1 + (change_magnitude_gene - 0.5) * 0.1, 0.001, 0.5);
    } else {
        candidate.target_type_str = "architectural_link_conceptual_cpp";
        candidate.target_identifier_str = "Link_ModuleA_ModuleB_Weight_sim";
        candidate.original_value_variant = 0.5; // Placeholder
        candidate.mutated_value_variant = std::clamp(0.5 + (change_magnitude_gene - 0.5) * 0.6, 0.0, 1.0);
    }

    candidate.mutation_operator_used_str = (operator_gene < 0.5) ? "gaussian_perturbation_sim_cpp" : "direct_value_set_sim_cpp";
    
    if(core_recombinator_ && uniform_dist_mugen_01_(rng_mugen_internal_) < 0.1 ) { // Log some translations for debug
        core_recombinator_->log_message("DEBUG", "MuGenCpp", 
            "Translated Genome to: Target=" + candidate.target_identifier_str + 
            ", OrigV=" + mugen_param_value_to_string(candidate.original_value_variant) + 
            ", MutV=" + mugen_param_value_to_string(candidate.mutated_value_variant) +
            ", Op=" + candidate.mutation_operator_used_str);
    }
    return candidate;
}

std::map<std::string, double> ControlledMutationGenerator::predict_impact_with_surrogates_internal_stub(
    const MutationCandidateCpp& candidate_to_evaluate,
    const GlobalSelfStateCpp& gs_context_for_prediction_sim) const {
    
    std::map<std::string, double> predicted_impacts_map;
    if (!core_recombinator_) return predicted_impacts_map;

    // STUB: Simulate impact prediction using the conceptual surrogate models.
    // A real implementation would involve feeding features of the candidate and context
    // into the actual surrogate models (which might be ML models from LearningModule).

    // Example: Impact on Coherence Score
    if (surrogate_models_internal_stub_.count("gs_coherence_impact_sim_model")) {
        // Conceptual: if the model is a vector of weights for genome features (as in init)
        // This is a placeholder, as the candidate is already concrete.
        // A better surrogate would take features of the *concrete candidate* and *context*.
        // For now, a random impact based on mutation type.
        double delta_coherence_sim = 0.0;
        if (candidate_to_evaluate.target_type_str.find("architectural") != std::string::npos) {
            delta_coherence_sim = (uniform_dist_mugen_01_(rng_mugen_internal_) - 0.6) * 0.15; // Architectural changes are risky
        } else if (std::holds_alternative<double>(candidate_to_evaluate.mutated_value_variant) &&
                   std::holds_alternative<double>(candidate_to_evaluate.original_value_variant)) {
            double change = std::get<double>(candidate_to_evaluate.mutated_value_variant) - 
                            std::get<double>(candidate_to_evaluate.original_value_variant);
            // Assume mutations to system params might be slightly destabilizing on average
            delta_coherence_sim = -std::abs(change) * 0.1 + (uniform_dist_mugen_01_(rng_mugen_internal_) - 0.5) * 0.02;
        }
        predicted_impacts_map["coherence_score_delta_sim_val"] = std::clamp(delta_coherence_sim, -0.2, 0.2);
    }

    // Example: Impact on System Entropy
    if (surrogate_models_internal_stub_.count("gs_entropy_impact_sim_model")) {
        double delta_entropy_sim = (uniform_dist_mugen_01_(rng_mugen_internal_) - 0.5) * 0.05; // Small random change
        if (candidate_to_evaluate.target_type_str.find("architectural") != std::string::npos) {
             delta_entropy_sim = (uniform_dist_mugen_01_(rng_mugen_internal_) - 0.4) * 0.1; // Arch changes might increase entropy
        }
        predicted_impacts_map["system_entropy_delta_sim_val"] = std::clamp(delta_entropy_sim, -0.1, 0.1);
    }
    
    // Example: Impact on Phi Funcional Score
    if (surrogate_models_internal_stub_.count("gs_phi_functional_impact_sim_model")) {
         predicted_impacts_map["phi_funcional_score_delta_sim_val"] = (uniform_dist_mugen_01_(rng_mugen_internal_) - 0.45) * 0.1; // Small, possibly positive
    }

    return predicted_impacts_map;
}

double ControlledMutationGenerator::calculate_overall_desirability_internal_stub(
    const std::map<std::string, double>& predicted_impact_vector,
    const std::string& fitness_landscape_id_from_sem_sim) const {
    
    double desirability_score = 0.0;
    if (!core_recombinator_) return desirability_score;

    // STUB: Calculate desirability based on current fitness landscape objectives.
    // This requires access to the FitnessLandscapeConfigCpp corresponding to fitness_landscape_id_from_sem_sim.
    // For this stub, we'll use hardcoded conceptual weights if SEM's landscape is not directly accessible here.
    // A better approach would be for SEM to pass its active_fitness_landscape or its objectives.

    // Conceptual weights for impact deltas (these should align with SEM's landscape goals)
    // Example: if SEM maximizes coherence, a positive coherence_delta is desirable.
    // If SEM targets low entropy, a negative entropy_delta (if current > target) or small delta is good.
    std::map<std::string, double> impact_delta_weights = {
        {"coherence_score_delta_sim_val", 0.5},      // Positive delta in coherence is good
        {"system_entropy_delta_sim_val", -0.3},     // Negative delta in entropy (towards lower) is good
        {"phi_funcional_score_delta_sim_val", 0.4} // Positive delta in phi_funcional is good
    };

    for (const auto& impact_pair : predicted_impact_vector) {
        if (impact_delta_weights.count(impact_pair.first)) {
            desirability_score += impact_pair.second * impact_delta_weights.at(impact_pair.first);
        }
    }
    
    // Normalize desirability to a range like [-1, 1] or [0, 1]
    // Using tanh for [-1, 1] like in Python version
    return std::tanh(desirability_score * 2.0); // Factor 2.0 to spread values more across tanh
}

MutationCandidateCpp ControlledMutationGenerator::evaluate_specific_mutation_candidate_with_surrogate(
    const MutationCandidateCpp& concrete_mutation_template,
    const GlobalSelfStateCpp& current_gs_for_context) {
    
    MutationCandidateCpp evaluated_candidate = concrete_mutation_template; // Start with a copy
    if (!core_recombinator_) {
        evaluated_candidate.overall_predicted_desirability_sim_val = -1.0; // Indicate error
        return evaluated_candidate;
    }
    evaluated_candidate.timestamp_val = core_recombinator_->get_current_timestamp();

    evaluated_candidate.predicted_impact_vector_sim_map = predict_impact_with_surrogates_internal_stub(
                                                            evaluated_candidate, current_gs_for_context);
    
    // Assume evaluation uses a "general" or "current SEM" fitness landscape context.
    // For simplicity, use a generic landscape ID for desirability calculation here.
    std::string eval_fitness_landscape_id = "general_evaluation_landscape_cpp";
    if (core_recombinator_ && core_recombinator_->get_global_state_const().active_fitness_landscape_config_for_sem.has_value()){
        // This is conceptual, as active_fitness_landscape_config_for_sem is not directly a string ID in GS Cpp.
        // eval_fitness_landscape_id = ... get ID from gs.active_fitness_landscape_config_for_sem ...;
    }


    evaluated_candidate.overall_predicted_desirability_sim_val = calculate_overall_desirability_internal_stub(
                                                                    evaluated_candidate.predicted_impact_vector_sim_map, eval_fitness_landscape_id);
    
    // Calculate aggregate confidence (same logic as in generate_and_propose...)
    double total_confidence_sum = 0.0; int num_confidences = 0;
    for(const auto& impact_pair : evaluated_candidate.predicted_impact_vector_sim_map){
        if (surrogate_model_confidence_internal_stub_.count("gs_coherence_impact_sim_model") && impact_pair.first.find("coherence") != std::string::npos) {
            total_confidence_sum += surrogate_model_confidence_internal_stub_.at("gs_coherence_impact_sim_model"); num_confidences++;
        } // Add more...
    }
    evaluated_candidate.simulation_confidence_sim_val = (num_confidences > 0) ? (total_confidence_sum / num_confidences) : 0.45; // Slightly lower default if no specific confidences

    evaluated_candidate.meets_improvement_threshold_sim_flag = 
        evaluated_candidate.overall_predicted_desirability_sim_val >= desirability_threshold_for_proposal_;

    if (core_recombinator_) {
        core_recombinator_->log_message("DEBUG", "MuGenCpp", 
            "Evaluated Specific Mutation: ID=" + evaluated_candidate.candidate_id +
            ", Target=" + evaluated_candidate.target_identifier_str +
            (evaluated_candidate.parameter_name_opt_str ? "." + *evaluated_candidate.parameter_name_opt_str : "") +
            ", OrigV=" + mugen_param_value_to_string(evaluated_candidate.original_value_variant) +
            ", MutV=" + mugen_param_value_to_string(evaluated_candidate.mutated_value_variant) +
            ", Desirability=" + std::to_string(evaluated_candidate.overall_predicted_desirability_sim_val));
    }
    return evaluated_candidate;
}
// eane_cpp_modules/lyuk_parser/lyuk_parser.cpp
// (Continuation from Part 7)
#include "lyuk_parser.h" // Already included
#include <sstream>   // For std::istringstream for line-by-line parsing
#include <algorithm> // For std::transform (to_upper), std::remove for trim
#include <iostream>  // For logging if core is null

// --- Parsing (LMI Phase 1 - Syntactic Parsing) ---
std::vector<std::shared_ptr<LyukBasePrimitiveASTNodeCpp>>
LyukParser::parse_lyuk_code_to_ast_stub(const std::string& lyuk_code_str) {
    std::vector<std::shared_ptr<LyukBasePrimitiveASTNodeCpp>> ast_sequence_vec;
    if (!core_recombinator_ && lyuk_code_str.empty()) return ast_sequence_vec; // No core for logging, or no code

    std::istringstream code_stream(lyuk_code_str);
    std::string current_line_str;
    int line_number = 1;

    std::string multi_line_buffer_str; // For primitives spanning multiple lines (conceptual)
    bool in_multi_line_primitive = false; // e.g., inside a FRAME block

    while (std::getline(code_stream, current_line_str)) {
        // Basic preprocessing: remove comments and trim whitespace
        size_t comment_pos = current_line_str.find("//"); // Lyuk might use different comment char
        if (comment_pos == std::string::npos) comment_pos = current_line_str.find('#'); // Python style
        if (comment_pos != std::string::npos) {
            current_line_str = current_line_str.substr(0, comment_pos);
        }
        current_line_str.erase(0, current_line_str.find_first_not_of(" \t\n\r\f\v"));
        current_line_str.erase(current_line_str.find_last_not_of(" \t\n\r\f\v") + 1);

        if (current_line_str.empty()) {
            line_number++;
            continue;
        }
        
        // STUB: Multi-line primitive handling is very complex.
        // For now, assume one primitive per processed line.
        // A real parser would use tokens like '{' '}' or 'ENDFRAME' to manage blocks.
        
        auto ast_node_ptr = parse_single_lyuk_primitive_line_stub(current_line_str, line_number);
        if (ast_node_ptr) {
            ast_sequence_vec.push_back(ast_node_ptr);
        } else if (core_recombinator_) { // Log parse failure for the line
            core_recombinator_->log_message("WARNING", "LyukParserCpp",
                "Failed to parse Lyuk line " + std::to_string(line_number) + ": '" + current_line_str.substr(0,50) + "'");
            auto error_node = std::make_shared<LyukBasePrimitiveASTNodeCpp>("PARSE_ERROR_CPP");
            error_node->raw_arguments_original_text_str = "Failed: " + current_line_str;
            error_node->source_line_ref_num = line_number;
            ast_sequence_vec.push_back(error_node);
        }
        line_number++;
    }
    return ast_sequence_vec;
}

std::shared_ptr<LyukBasePrimitiveASTNodeCpp> LyukParser::parse_single_lyuk_primitive_line_stub(
    const std::string& lyuk_line_str, int line_num) {
    
    std::string primitive_name = extract_primitive_name_from_line_stub(lyuk_line_str);
    if (primitive_name.empty()) return nullptr;

    std::string primitive_name_upper = primitive_name;
    std::transform(primitive_name_upper.begin(), primitive_name_upper.end(), primitive_name_upper.begin(), ::toupper);

    std::shared_ptr<LyukBasePrimitiveASTNodeCpp> node_ptr;
    if (primitive_name_upper == "FRAME") node_ptr = std::make_shared<LyukFrameASTNodeCpp>();
    else if (primitive_name_upper == "LET") node_ptr = std::make_shared<LyukLetASTNodeCpp>();
    // ... add more else if for other specific Lyuk AST node types ...
    else node_ptr = std::make_shared<LyukBasePrimitiveASTNodeCpp>(primitive_name_upper);
    
    node_ptr->raw_arguments_original_text_str = extract_raw_arguments_from_line_stub(lyuk_line_str);
    node_ptr->source_line_ref_num = line_num;
    node_ptr->parsed_arguments_vec = parse_arguments_stub(node_ptr->raw_arguments_original_text_str);

    // Special handling for LET to extract destination_variable_name
    if (primitive_name_upper == "LET" && node_ptr->parsed_arguments_vec.size() >= 1) {
        // Assume format: LET VarName ...
        // The first "argument" after LET is the variable name.
        // The parse_arguments_stub needs to be smart enough to handle this.
        // For this stub, let's assume the first token in raw_arguments is the var name.
        std::string raw_args = node_ptr->raw_arguments_original_text_str;
        std::string var_name_token;
        std::istringstream temp_arg_stream(raw_args);
        if (temp_arg_stream >> var_name_token) { // Get first token
            node_ptr->destination_variable_name_opt_str = var_name_token;
            // The actual value assignment part would be the rest of raw_args or subsequent parsed_arguments.
        }
    }
    return node_ptr;
}

std::string LyukParser::extract_primitive_name_from_line_stub(const std::string& lyuk_line_str) const {
    std::istringstream iss(lyuk_line_str);
    std::string first_token;
    iss >> first_token; // Reads the first word separated by whitespace
    return first_token;
}

std::string LyukParser::extract_raw_arguments_from_line_stub(const std::string& lyuk_line_str) const {
    size_t first_space_pos = lyuk_line_str.find_first_of(" \t");
    if (first_space_pos == std::string::npos || first_space_pos == lyuk_line_str.length() - 1) {
        return ""; // No arguments or only primitive name
    }
    std::string args_part = lyuk_line_str.substr(first_space_pos + 1);
    // Trim leading whitespace from args_part
    args_part.erase(0, args_part.find_first_not_of(" \t"));
    return args_part;
}

// STUB for argument parsing. A real implementation needs a proper tokenizer/parser for Lyuk syntax.
std::vector<std::variant<std::shared_ptr<LyukBasePrimitiveASTNodeCpp>, LyukDataTypeCpp>>
LyukParser::parse_arguments_stub(const std::string& raw_args_str) {
    std::vector<std::variant<std::shared_ptr<LyukBasePrimitiveASTNodeCpp>, LyukDataTypeCpp>> parsed_args_vec;
    if (raw_args_str.empty()) return parsed_args_vec;

    std::istringstream args_stream(raw_args_str);
    std::string token;
    
    // This stub just splits by space and tries to guess type for simple literals.
    // It doesn't handle nested structures, quotes properly, operators, etc.
    while (args_stream >> token) {
        if (token.empty()) continue;
        LyukDataTypeCpp arg_data;
        arg_data.is_literal_flag = true;

        if (token == "true" || token == "TRUE") {
            arg_data.type_name_str = "Boolean"; arg_data.value_variant = true;
        } else if (token == "false" || token == "FALSE") {
            arg_data.type_name_str = "Boolean"; arg_data.value_variant = false;
        } else if (token.front() == '"' && token.back() == '"' && token.length() >= 2) {
            arg_data.type_name_str = "String"; arg_data.value_variant = token.substr(1, token.length() - 2);
        } else {
            // Try to parse as number (double then long long)
            try {
                size_t processed_chars_double = 0;
                double d_val = std::stod(token, &processed_chars_double);
                if (processed_chars_double == token.length()) { // Entire token was a double
                    arg_data.type_name_str = "Number"; arg_data.value_variant = d_val;
                    parsed_args_vec.push_back(arg_data); continue;
                }
            } catch (const std::exception&) { /* Not a double or only partially */ }
            try {
                size_t processed_chars_long = 0;
                long long ll_val = std::stoll(token, &processed_chars_long);
                if (processed_chars_long == token.length()) { // Entire token was a long long
                    arg_data.type_name_str = "Number"; arg_data.value_variant = ll_val;
                    parsed_args_vec.push_back(arg_data); continue;
                }
            } catch (const std::exception&) { /* Not a long long */ }
            
            // Default to Symbol if not a recognized literal
            arg_data.type_name_str = "Symbol"; arg_data.value_variant = token;
        }
        parsed_args_vec.push_back(arg_data);
    }
    return parsed_args_vec;
}

// --- Semantic Interpretation (LMI Phase 2 - Stub) ---
std::map<std::string, std::any> LyukParser::interpret_ast_semantically_stub(
    const std::vector<std::shared_ptr<LyukBasePrimitiveASTNodeCpp>>& ast_sequence) {
    
    std::map<std::string, std::any> interpretation_summary_map;
    if (!core_recombinator_) {
        interpretation_summary_map["error_str"] = std::string("Core not available for semantic interpretation.");
        return interpretation_summary_map;
    }
    core_recombinator_->log_message("DEBUG", "LyukParserCpp", 
        "STUB: Performing semantic interpretation on AST with " + std::to_string(ast_sequence.size()) + " root nodes.");

    int let_count = 0;
    int felt_count = 0;
    std::vector<std::string> variable_names_declared_sim_vec;

    for (const auto& node_ptr : ast_sequence) {
        if (!node_ptr) continue;
        if (node_ptr->primitive_name_str == "LET") {
            let_count++;
            if (node_ptr->destination_variable_name_opt_str) {
                variable_names_declared_sim_vec.push_back(*node_ptr->destination_variable_name_opt_str);
            }
        } else if (node_ptr->primitive_name_str == "FELT") {
            felt_count++;
        }
        // A real semantic analyzer would:
        // - Build and manage symbol tables / scopes.
        // - Perform type checking on arguments and expressions.
        // - Resolve symbol references.
        // - Potentially build a control flow graph or intermediate representation.
        // - Infer intentions, emotions, logical assertions from the AST.
    }
    interpretation_summary_map["num_let_statements_sim_val"] = let_count;
    interpretation_summary_map["num_felt_statements_sim_val"] = felt_count;
    // interpretation_summary_map["declared_variables_sample_vec_str"] = variable_names_declared_sim_vec; // std::any can't take vector directly easily for Pybind

    return interpretation_summary_map;
}

// --- Transcompilation (LTC - Stub) ---
std::string LyukParser::transcompile_ast_to_human_readable_stub(
    const std::vector<std::shared_ptr<LyukBasePrimitiveASTNodeCpp>>& ast_sequence,
    const std::string& target_representation_type) {
    
    std::ostringstream oss;
    oss << "// Lyuk AST Transcompiled to: " << target_representation_type << " (C++ Stub)\n";
    oss << "// Total top-level primitives: " << ast_sequence.size() << "\n---\n";

    std::function<void(const std::shared_ptr<LyukBasePrimitiveASTNodeCpp>&, int)> transpile_node_recursive;
    transpile_node_recursive = 
        [&](const std::shared_ptr<LyukBasePrimitiveASTNodeCpp>& node_ptr, int indent_level) {
        if (!node_ptr) return;
        std::string indent_str(indent_level * 2, ' '); // 2 spaces per indent level

        oss << indent_str << "/* Line " << node_ptr->source_line_ref_num << " */ ";
        oss << node_ptr->primitive_name_str;

        if (!node_ptr->parsed_arguments_vec.empty()) {
            oss << "(";
            for (size_t i = 0; i < node_ptr->parsed_arguments_vec.size(); ++i) {
                const auto& arg_variant = node_ptr->parsed_arguments_vec[i];
                if (std::holds_alternative<LyukDataTypeCpp>(arg_variant)) {
                    const auto& data_val = std::get<LyukDataTypeCpp>(arg_variant);
                    oss << mugen_param_value_to_string(data_val.value_variant); // Use helper (if it matches LyukValueVariantCpp)
                                                                              // Or a dedicated to_string for LyukValueVariantCpp
                } else if (std::holds_alternative<std::shared_ptr<LyukBasePrimitiveASTNodeCpp>>(arg_variant)) {
                    oss << "<NestedAST:" << std::get<std::shared_ptr<LyukBasePrimitiveASTNodeCpp>>(arg_variant)->primitive_name_str << ">";
                }
                if (i < node_ptr->parsed_arguments_vec.size() - 1) oss << ", ";
            }
            oss << ")";
        }
        
        if (node_ptr->destination_variable_name_opt_str) {
            oss << " => " << *node_ptr->destination_variable_name_opt_str;
        }
        oss << ";\n";

        // If it's a block-like node (e.g., FRAME, IF, LOOP), recursively transpile its body
        if (auto frame_node = std::dynamic_pointer_cast<LyukFrameASTNodeCpp>(node_ptr)) {
            oss << indent_str << "{\n";
            for (const auto& body_node_ptr : frame_node->body_primitives_vec) {
                transpile_node_recursive(body_node_ptr, indent_level + 1);
            }
            oss << indent_str << "}\n";
        }
        // ... Add similar for IF, LOOP bodies ...
    };

    for (const auto& root_node_ptr : ast_sequence) {
        transpile_node_recursive(root_node_ptr, 0);
    }
    oss << "---\n// End of Transcompilation Stub\n";
    return oss.str();
}
// eane_cpp_modules/consciousness_module_cpp/consciousness_module.h
#pragma once
#include "../core_interface.h"
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <deque>    // For phi_history_short_term_
#include <map>      // For active_mental_experiments_
#include <optional> // For optional returns or members

// C++ equivalent of Python's ConsciousState dataclass
struct ConsciousStateCpp {
    Eigen::VectorXd perception_vec; // Renamed
    Eigen::VectorXd decision_vec;   // Renamed
    Eigen::VectorXd narrative_vec;  // Renamed

    ConsciousStateCpp(int p_dim, int d_dim, int n_dim) :
        perception_vec(Eigen::VectorXd::Zero(p_dim > 0 ? p_dim : 0)),
        decision_vec(Eigen::VectorXd::Zero(d_dim > 0 ? d_dim : 0)),
        narrative_vec(Eigen::VectorXd::Zero(n_dim > 0 ? n_dim : 0)) {}
};

// C++ equivalent of Python's MentalExperimentLog dataclass (simplified)
struct MentalExperimentLogCpp {
    std::string experiment_id_str;
    std::string creator_query_str;
    double start_timestamp_val = 0.0;
    std::string status_str = "initiated_cpp_exp";
    std::vector<std::string> involved_modules_conceptual_vec;
    // std::vector<std::map<std::string, std::any>> key_insights_generated_vec_sim; // std::any is complex
    std::optional<double> impact_on_phi_observed_opt_val;
    std::optional<std::string> shimyureshon_reflexion_id_opt_str;
    // ... more fields as needed, matching Python version where feasible ...
};


class ConsciousnessModuleCpp {
public:
    ConsciousnessModuleCpp(CoreInterface* core,
                           int perception_dim = 10, 
                           int decision_dim = 3, 
                           int narrative_dim = 5,
                           double phi_modulation_factor = 0.15);
    
    void update_logic();

    // Method to be called by Core when Shimyureshon (for mental experiment) results are ready
    // report_sim: A map representing the ShimyureshonMetricsReport (simplified for C++)
    void process_shimyureshon_reflexion_results_stub(const std::string& sh_id, 
                                                     const std::map<std::string, std::any>& report_sim);

private:
    CoreInterface* core_recombinator_; // Non-owning
    int perception_dim_;
    int decision_dim_;
    int narrative_dim_;
    double phi_modulation_factor_;

    ConsciousStateCpp internal_conscious_state_obj_; // Renamed
    Eigen::MatrixXd W_n_matrix_; // Narrative weights (narrative_dim x (perception_dim + decision_dim))
    Eigen::MatrixXd W_util_matrix_; // Utility/decision weights (perception_dim x decision_dim)

    std::deque<double> phi_history_short_term_deque_; // Maxlen managed manually or by deque if limited
    const size_t phi_history_maxlen_ = 20;

    std::map<std::string, MentalExperimentLogCpp> active_mental_experiments_map_;

    // --- Internal Helper Methods ---
    Eigen::VectorXd make_internal_decision_cm_internal(const Eigen::VectorXd& current_perception_vec); // Renamed
    Eigen::VectorXd build_internal_narrative_cm_internal(const Eigen::VectorXd& current_perception_vec, 
                                                      const Eigen::VectorXd& current_decision_vec); // Renamed
    double compute_phi_detailed_cm_stub() const; // Renamed with _stub

    // Methods for mental experiments (stubs)
    void process_creator_mental_experiment_request_stub();
    void launch_shimyureshon_for_mental_experiment_stub(const std::string& experiment_id, const std::string& query, const std::string& profile_key);
};
// eane_cpp_modules/consciousness_module_cpp/consciousness_module.cpp
// (Continuation from Part 7)
#include "consciousness_module.h" // Already included

void ConsciousnessModuleCpp::process_shimyureshon_reflexion_results_stub(
    const std::string& sh_id, 
    const std::map<std::string, std::any>& report_sim) {
    
    if (!core_recombinator_) return;
    core_recombinator_->log_message("INFO", "ConsciousnessModuleCpp", "STUB: Processing Shimyureshon results for SH_ID: " + sh_id);

    std::string experiment_id_found;
    auto it_exp = std::find_if(active_mental_experiments_map_.begin(), active_mental_experiments_map_.end(),
        [&sh_id](const auto& pair_exp) {
            return pair_exp.second.shimyureshon_reflexion_id_opt_str.has_value() &&
                   *(pair_exp.second.shimyureshon_reflexion_id_opt_str) == sh_id;
        });

    if (it_exp == active_mental_experiments_map_.end()) {
        core_recombinator_->log_message("WARNING", "ConsciousnessModuleCpp", "No active mental experiment found for Shimyureshon ID: " + sh_id);
        return;
    }
    MentalExperimentLogCpp& log_entry = it_exp->second;
    experiment_id_found = it_exp->first;

    // STUB: Extract relevant metrics from report_sim (which is std::any map)
    // This requires careful casting.
    double sh_success_score_sim = 0.7; // Placeholder
    std::string sh_summary_insight_sim = "Shimyureshon provided conceptual insights.";
    // if (report_sim.count("custom_scenario_metrics_map_any")) { ... cast and extract ... }

    log_entry.status_str = "reflection_generated_from_sh_cpp";
    // log_entry.resolution_summary_str = sh_summary_insight_sim;
    // log_entry.understanding_depth_score_calc_val = sh_success_score_sim * 0.8; // Conceptual
    log_entry.impact_on_phi_observed_opt_val = core_recombinator_->get_global_state_const().phi_consciousness;
    // log_entry.completion_timestamp_val = core_recombinator_->get_current_timestamp();

    core_recombinator_->log_message("INFO", "ConsciousnessModuleCpp", 
        "Mental experiment '" + log_entry.experiment_id_str + "' (from SH: " + sh_id + ") processing completed (stub).");

    // Send event to NarrativeSelfCpp (conceptual)
    // EventDataCpp ns_event("mental_experiment_resolution_fused_for_ns_cpp");
    // ... populate content ...
    // core_recombinator_->event_queue_put(ns_event);

    active_mental_experiments_map_.erase(experiment_id_found); // Remove completed experiment
}


// STUB for launching Shimyureshon
void ConsciousnessModuleCpp::launch_shimyureshon_for_mental_experiment_stub(
    const std::string& experiment_id, const std::string& query, const std::string& profile_key) {
    if (!core_recombinator_) return;
    core_recombinator_->log_message("INFO", "ConsciousnessModuleCpp", 
        "STUB: Requesting Shimyureshon for experiment ID: " + experiment_id + ", Profile: " + profile_key);
    
    if (active_mental_experiments_map_.count(experiment_id)) {
        active_mental_experiments_map_[experiment_id].shimyureshon_reflexion_id_opt_str = "sh_sim_id_" + experiment_id;
        active_mental_experiments_map_[experiment_id].status_str = "shimyureshon_running_cpp_sim";
    }
    // In a real system, this would prepare parameters and call core_recombinator_->start_shimyureshon_cpp(...)
    // For the stub, we can simulate that it starts and will eventually complete.
    // A callback or polling mechanism would be needed for results.
}

// STUB for processing mental experiment requests
void ConsciousnessModuleCpp::process_creator_mental_experiment_request_stub() {
    if(!core_recombinator_) return;
    // Conceptual: Check event queue for "creator_mental_experiment_input_cpp"
    // EventDataCpp exp_request_event;
    // bool found = core_recombinator_->event_queue_get_specific_cpp(
    //                  "creator_mental_experiment_input_cpp", exp_request_event, 0.001);
    // if (found) {
    //     std::string exp_id = "mexp_cpp_default";
    //     std::string query_str = "Default C++ query";
    //     std::string profile_key_str = "default_deep_dive_cpp";
    //     if(exp_request_event.content.count("experiment_id_str")) { try{ exp_id = std::get<std::string>(exp_request_event.content.at("experiment_id_str")); } catch(...){} }
    //     if(exp_request_event.content.count("query_str_val")) { try{ query_str = std::get<std::string>(exp_request_event.content.at("query_str_val")); } catch(...){} }
    //     // ... get profile_key ...
    //
    //     if (active_mental_experiments_map_.find(exp_id) == active_mental_experiments_map_.end()) {
    //         MentalExperimentLogCpp new_log;
    //         new_log.experiment_id_str = exp_id;
    //         new_log.creator_query_str = query_str;
    //         new_log.start_timestamp_val = core_recombinator_->get_current_timestamp();
    //         active_mental_experiments_map_[exp_id] = new_log;
    //         launch_shimyureshon_for_mental_experiment_stub(exp_id, query_str, profile_key_str);
    //     }
    // }
}
```cpp
// eane_cpp_modules/controlled_mutation_generator/controlled_mutation_generator.cpp
// (Continuation from Part 10, previous response)
#include "controlled_mutation_generator.h" // Already included

MutationCandidateCpp ControlledMutationGenerator::translate_abstract_genome_to_concrete_mutation_internal_stub(
    const Eigen::VectorXd& abstract_genome,
    const std::string& fitness_landscape_id_from_sem_sim,
    const GlobalSelfStateCpp& gs_context_for_translation_sim) const {

    MutationCandidateCpp candidate; // ID is auto-generated
    candidate.tags_vec.push_back("from_abstract_genome_cpp_stub");
    candidate.timestamp_val = core_recombinator_ ? core_recombinator_->get_current_timestamp() : 0.0;

    if (abstract_genome.size() == 0 || (abstract_genome_dim_hint_ > 0 && abstract_genome.size() != abstract_genome_dim_hint_)) {
        if(core_recombinator_) core_recombinator_->log_message("ERROR", "MuGenCpp", "TranslateGenome: Genome size mismatch in stub.");
        candidate.target_type_str = "translation_error_cpp";
        return candidate;
    }

    // STUB: Very simplified translation logic.
    // A real system would have complex rules or a learned model to map genome to concrete mutations.
    int gene_idx = 0;
    auto get_gene_val = [&](bool increment = true) { // Helper to get gene and advance index safely
        double val = 0.5; // Default
        if(abstract_genome.size() > 0) val = abstract_genome(gene_idx % abstract_genome.size());
        if(increment) gene_idx++;
        return val;
    };

    double target_type_gene = get_gene_val();
    double target_param_gene = get_gene_val();
    double operator_gene = get_gene_val();
    double change_magnitude_gene = get_gene_val(); // In [0,1]

    if (target_type_gene < 0.4) {
        candidate.target_type_str = "system_parameter_gs_cpp";
        std::vector<std::string> gs_params = {"GlobalSelfStateCpp.coherence_score", "GlobalSelfStateCpp.system_entropy", "GlobalSelfStateCpp.motivacion", "GlobalSelfStateCpp.phi_funcional_score"};
        candidate.target_identifier_str = gs_params[static_cast<int>(target_param_gene * gs_params.size()) % gs_params.size()];
        
        // Get original value (conceptual)
        if (candidate.target_identifier_str == "GlobalSelfStateCpp.coherence_score") candidate.original_value_variant = gs_context_for_translation_sim.coherence_score;
        else if (candidate.target_identifier_str == "GlobalSelfStateCpp.system_entropy") candidate.original_value_variant = gs_context_for_translation_sim.system_entropy;
        else if (candidate.target_identifier_str == "GlobalSelfStateCpp.motivacion") candidate.original_value_variant = gs_context_for_translation_sim.motivacion;
        else if (candidate.target_identifier_str == "GlobalSelfStateCpp.phi_funcional_score") candidate.original_value_variant = gs_context_for_translation_sim.phi_funcional_score;
        else candidate.original_value_variant = 0.5; // Default original

        double current_val = std::holds_alternative<double>(candidate.original_value_variant) ? std::get<double>(candidate.original_value_variant) : 0.5;
        double change = (change_magnitude_gene - 0.5) * 0.2; // Max change +/- 0.1
        candidate.mutated_value_variant = std::clamp(current_val + change, 0.0, 1.0);

    } else if (target_type_gene < 0.8) {
        candidate.target_type_str = "module_parameter_runtime_cpp";
        // Example module parameters (these would need to be actual accessible parameters)
        std::vector<std::string> mod_params = {"LearningModuleCpp.alpha_lr_fwe_stub", "SelfEvolutionModuleCpp.mutation_rate_base_stub", "EmotionRegulationModuleCpp.kp_erm_stub"};
        candidate.target_identifier_str = mod_params[static_cast<int>(target_param_gene * mod_params.size()) % mod_params.size()];
        candidate.original_value_variant = 0.1; // Placeholder original
        candidate.mutated_value_variant = std::clamp(0.1 + (change_magnitude_gene - 0.5) * 0.1, 0.001, 0.5);
    } else {
        candidate.target_type_str = "architectural_link_conceptual_cpp";
        candidate.target_identifier_str = "Link_ModuleA_ModuleB_Weight_sim";
        candidate.original_value_variant = 0.5; // Placeholder
        candidate.mutated_value_variant = std::clamp(0.5 + (change_magnitude_gene - 0.5) * 0.6, 0.0, 1.0);
    }

    candidate.mutation_operator_used_str = (operator_gene < 0.5) ? "gaussian_perturbation_sim_cpp" : "direct_value_set_sim_cpp";
    
    if(core_recombinator_ && uniform_dist_mugen_01_(rng_mugen_internal_) < 0.1 ) { // Log some translations for debug
        core_recombinator_->log_message("DEBUG", "MuGenCpp", 
            "Translated Genome to: Target=" + candidate.target_identifier_str + 
            ", OrigV=" + mugen_param_value_to_string(candidate.original_value_variant) + 
            ", MutV=" + mugen_param_value_to_string(candidate.mutated_value_variant) +
            ", Op=" + candidate.mutation_operator_used_str);
    }
    return candidate;
}

std::map<std::string, double> ControlledMutationGenerator::predict_impact_with_surrogates_internal_stub(
    const MutationCandidateCpp& candidate_to_evaluate,
    const GlobalSelfStateCpp& gs_context_for_prediction_sim) const {
    
    std::map<std::string, double> predicted_impacts_map;
    if (!core_recombinator_) return predicted_impacts_map;

    // STUB: Simulate impact prediction using the conceptual surrogate models.
    // A real implementation would involve feeding features of the candidate and context
    // into the actual surrogate models (which might be ML models from LearningModule).

    // Example: Impact on Coherence Score
    if (surrogate_models_internal_stub_.count("gs_coherence_impact_sim_model")) {
        // Conceptual: if the model is a vector of weights for genome features (as in init)
        // This is a placeholder, as the candidate is already concrete.
        // A better surrogate would take features of the *concrete candidate* and *context*.
        // For now, a random impact based on mutation type.
        double delta_coherence_sim = 0.0;
        if (candidate_to_evaluate.target_type_str.find("architectural") != std::string::npos) {
            delta_coherence_sim = (uniform_dist_mugen_01_(rng_mugen_internal_) - 0.6) * 0.15; // Architectural changes are risky
        } else if (std::holds_alternative<double>(candidate_to_evaluate.mutated_value_variant) &&
                   std::holds_alternative<double>(candidate_to_evaluate.original_value_variant)) {
            double change = std::get<double>(candidate_to_evaluate.mutated_value_variant) - 
                            std::get<double>(candidate_to_evaluate.original_value_variant);
            // Assume mutations to system params might be slightly destabilizing on average
            delta_coherence_sim = -std::abs(change) * 0.1 + (uniform_dist_mugen_01_(rng_mugen_internal_) - 0.5) * 0.02;
        }
        predicted_impacts_map["coherence_score_delta_sim_val"] = std::clamp(delta_coherence_sim, -0.2, 0.2);
    }

    // Example: Impact on System Entropy
    if (surrogate_models_internal_stub_.count("gs_entropy_impact_sim_model")) {
        double delta_entropy_sim = (uniform_dist_mugen_01_(rng_mugen_internal_) - 0.5) * 0.05; // Small random change
        if (candidate_to_evaluate.target_type_str.find("architectural") != std::string::npos) {
             delta_entropy_sim = (uniform_dist_mugen_01_(rng_mugen_internal_) - 0.4) * 0.1; // Arch changes might increase entropy
        }
        predicted_impacts_map["system_entropy_delta_sim_val"] = std::clamp(delta_entropy_sim, -0.1, 0.1);
    }
    
    // Example: Impact on Phi Funcional Score
    if (surrogate_models_internal_stub_.count("gs_phi_functional_impact_sim_model")) {
         predicted_impacts_map["phi_funcional_score_delta_sim_val"] = (uniform_dist_mugen_01_(rng_mugen_internal_) - 0.45) * 0.1; // Small, possibly positive
    }

    return predicted_impacts_map;
}

double ControlledMutationGenerator::calculate_overall_desirability_internal_stub(
    const std::map<std::string, double>& predicted_impact_vector,
    const std::string& fitness_landscape_id_from_sem_sim) const {
    
    double desirability_score = 0.0;
    if (!core_recombinator_) return desirability_score;

    // STUB: Calculate desirability based on current fitness landscape objectives.
    // This requires access to the FitnessLandscapeConfigCpp corresponding to fitness_landscape_id_from_sem_sim.
    // For this stub, we'll use hardcoded conceptual weights if SEM's landscape is not directly accessible here.
    // A better approach would be for SEM to pass its active_fitness_landscape or its objectives.

    // Conceptual weights for impact deltas (these should align with SEM's landscape goals)
    // Example: if SEM maximizes coherence, a positive coherence_delta is desirable.
    // If SEM targets low entropy, a negative entropy_delta (if current > target) or small delta is good.
    std::map<std::string, double> impact_delta_weights = {
        {"coherence_score_delta_sim_val", 0.5},      // Positive delta in coherence is good
        {"system_entropy_delta_sim_val", -0.3},     // Negative delta in entropy (towards lower) is good
        {"phi_funcional_score_delta_sim_val", 0.4} // Positive delta in phi_funcional is good
    };

    for (const auto& impact_pair : predicted_impact_vector) {
        if (impact_delta_weights.count(impact_pair.first)) {
            desirability_score += impact_pair.second * impact_delta_weights.at(impact_pair.first);
        }
    }
    
    // Normalize desirability to a range like [-1, 1] or [0, 1]
    // Using tanh for [-1, 1] like in Python version
    return std::tanh(desirability_score * 2.0); // Factor 2.0 to spread values more across tanh
}

MutationCandidateCpp ControlledMutationGenerator::evaluate_specific_mutation_candidate_with_surrogate(
    const MutationCandidateCpp& concrete_mutation_template,
    const GlobalSelfStateCpp& current_gs_for_context) {
    
    MutationCandidateCpp evaluated_candidate = concrete_mutation_template; // Start with a copy
    if (!core_recombinator_) {
        evaluated_candidate.overall_predicted_desirability_sim_val = -1.0; // Indicate error
        return evaluated_candidate;
    }
    evaluated_candidate.timestamp_val = core_recombinator_->get_current_timestamp();

    evaluated_candidate.predicted_impact_vector_sim_map = predict_impact_with_surrogates_internal_stub(
                                                            evaluated_candidate, current_gs_for_context);
    
    // Assume evaluation uses a "general" or "current SEM" fitness landscape context.
    // For simplicity, use a generic landscape ID for desirability calculation here.
    std::string eval_fitness_landscape_id = "general_evaluation_landscape_cpp";
    if (core_recombinator_ && core_recombinator_->get_global_state_const().active_fitness_landscape_config_for_sem.has_value()){
        // This is conceptual, as active_fitness_landscape_config_for_sem is not directly a string ID in GS Cpp.
        // eval_fitness_landscape_id = ... get ID from gs.active_fitness_landscape_config_for_sem ...;
    }


    evaluated_candidate.overall_predicted_desirability_sim_val = calculate_overall_desirability_internal_stub(
                                                                    evaluated_candidate.predicted_impact_vector_sim_map, eval_fitness_landscape_id);
    
    // Calculate aggregate confidence (same logic as in generate_and_propose...)
    double total_confidence_sum = 0.0; int num_confidences = 0;
    for(const auto& impact_pair : evaluated_candidate.predicted_impact_vector_sim_map){
        if (surrogate_model_confidence_internal_stub_.count("gs_coherence_impact_sim_model") && impact_pair.first.find("coherence") != std::string::npos) {
            total_confidence_sum += surrogate_model_confidence_internal_stub_.at("gs_coherence_impact_sim_model"); num_confidences++;
        } // Add more...
    }
    evaluated_candidate.simulation_confidence_sim_val = (num_confidences > 0) ? (total_confidence_sum / num_confidences) : 0.45; // Slightly lower default if no specific confidences

    evaluated_candidate.meets_improvement_threshold_sim_flag = 
        evaluated_candidate.overall_predicted_desirability_sim_val >= desirability_threshold_for_proposal_;

    if (core_recombinator_) {
        core_recombinator_->log_message("DEBUG", "MuGenCpp", 
            "Evaluated Specific Mutation: ID=" + evaluated_candidate.candidate_id +
            ", Target=" + evaluated_candidate.target_identifier_str +
            (evaluated_candidate.parameter_name_opt_str ? "." + *evaluated_candidate.parameter_name_opt_str : "") +
            ", OrigV=" + mugen_param_value_to_string(evaluated_candidate.original_value_variant) +
            ", MutV=" + mugen_param_value_to_string(evaluated_candidate.mutated_value_variant) +
            ", Desirability=" + std::to_string(evaluated_candidate.overall_predicted_desirability_sim_val));
    }
    return evaluated_candidate;
}
```

```cpp
// eane_cpp_modules/lyuk_parser/lyuk_parser.cpp
// (Continuation from Part 7)
#include "lyuk_parser.h" // Already included
#include <sstream>   // For std::istringstream for line-by-line parsing
#include <algorithm> // For std::transform (to_upper), std::remove for trim
#include <iostream>  // For logging if core is null

// --- Parsing (LMI Phase 1 - Syntactic Parsing) ---
std::vector<std::shared_ptr<LyukBasePrimitiveASTNodeCpp>>
LyukParser::parse_lyuk_code_to_ast_stub(const std::string& lyuk_code_str) {
    std::vector<std::shared_ptr<LyukBasePrimitiveASTNodeCpp>> ast_sequence_vec;
    if (!core_recombinator_ && lyuk_code_str.empty()) return ast_sequence_vec; // No core for logging, or no code

    std::istringstream code_stream(lyuk_code_str);
    std::string current_line_str;
    int line_number = 1;

    std::string multi_line_buffer_str; // For primitives spanning multiple lines (conceptual)
    bool in_multi_line_primitive = false; // e.g., inside a FRAME block

    while (std::getline(code_stream, current_line_str)) {
        // Basic preprocessing: remove comments and trim whitespace
        size_t comment_pos = current_line_str.find("//"); // Lyuk might use different comment char
        if (comment_pos == std::string::npos) comment_pos = current_line_str.find('#'); // Python style
        if (comment_pos != std::string::npos) {
            current_line_str = current_line_str.substr(0, comment_pos);
        }
        current_line_str.erase(0, current_line_str.find_first_not_of(" \t\n\r\f\v"));
        current_line_str.erase(current_line_str.find_last_not_of(" \t\n\r\f\v") + 1);

        if (current_line_str.empty()) {
            line_number++;
            continue;
        }
        
        // STUB: Multi-line primitive handling is very complex.
        // For now, assume one primitive per processed line.
        // A real parser would use tokens like '{' '}' or 'ENDFRAME' to manage blocks.
        
        auto ast_node_ptr = parse_single_lyuk_primitive_line_stub(current_line_str, line_number);
        if (ast_node_ptr) {
            ast_sequence_vec.push_back(ast_node_ptr);
        } else if (core_recombinator_) { // Log parse failure for the line
            core_recombinator_->log_message("WARNING", "LyukParserCpp",
                "Failed to parse Lyuk line " + std::to_string(line_number) + ": '" + current_line_str.substr(0,50) + "'");
            auto error_node = std::make_shared<LyukBasePrimitiveASTNodeCpp>("PARSE_ERROR_CPP");
            error_node->raw_arguments_original_text_str = "Failed: " + current_line_str;
            error_node->source_line_ref_num = line_number;
            ast_sequence_vec.push_back(error_node);
        }
        line_number++;
    }
    return ast_sequence_vec;
}

std::shared_ptr<LyukBasePrimitiveASTNodeCpp> LyukParser::parse_single_lyuk_primitive_line_stub(
    const std::string& lyuk_line_str, int line_num) {
    
    std::string primitive_name = extract_primitive_name_from_line_stub(lyuk_line_str);
    if (primitive_name.empty()) return nullptr;

    std::string primitive_name_upper = primitive_name;
    std::transform(primitive_name_upper.begin(), primitive_name_upper.end(), primitive_name_upper.begin(), ::toupper);

    std::shared_ptr<LyukBasePrimitiveASTNodeCpp> node_ptr;
    if (primitive_name_upper == "FRAME") node_ptr = std::make_shared<LyukFrameASTNodeCpp>();
    else if (primitive_name_upper == "LET") node_ptr = std::make_shared<LyukLetASTNodeCpp>();
    // ... add more else if for other specific Lyuk AST node types ...
    else node_ptr = std::make_shared<LyukBasePrimitiveASTNodeCpp>(primitive_name_upper);
    
    node_ptr->raw_arguments_original_text_str = extract_raw_arguments_from_line_stub(lyuk_line_str);
    node_ptr->source_line_ref_num = line_num;
    node_ptr->parsed_arguments_vec = parse_arguments_stub(node_ptr->raw_arguments_original_text_str);

    // Special handling for LET to extract destination_variable_name
    if (primitive_name_upper == "LET" && node_ptr->parsed_arguments_vec.size() >= 1) {
        // Assume format: LET VarName ...
        // The first "argument" after LET is the variable name.
        // The parse_arguments_stub needs to be smart enough to handle this.
        // For this stub, let's assume the first token in raw_arguments is the var name.
        std::string raw_args = node_ptr->raw_arguments_original_text_str;
        std::string var_name_token;
        std::istringstream temp_arg_stream(raw_args);
        if (temp_arg_stream >> var_name_token) { // Get first token
            node_ptr->destination_variable_name_opt_str = var_name_token;
            // The actual value assignment part would be the rest of raw_args or subsequent parsed_arguments.
        }
    }
    return node_ptr;
}

std::string LyukParser::extract_primitive_name_from_line_stub(const std::string& lyuk_line_str) const {
    std::istringstream iss(lyuk_line_str);
    std::string first_token;
    iss >> first_token; // Reads the first word separated by whitespace
    return first_token;
}

std::string LyukParser::extract_raw_arguments_from_line_stub(const std::string& lyuk_line_str) const {
    size_t first_space_pos = lyuk_line_str.find_first_of(" \t");
    if (first_space_pos == std::string::npos || first_space_pos == lyuk_line_str.length() - 1) {
        return ""; // No arguments or only primitive name
    }
    std::string args_part = lyuk_line_str.substr(first_space_pos + 1);
    // Trim leading whitespace from args_part
    args_part.erase(0, args_part.find_first_not_of(" \t"));
    return args_part;
}

// STUB for argument parsing. A real implementation needs a proper tokenizer/parser for Lyuk syntax.
std::vector<std::variant<std::shared_ptr<LyukBasePrimitiveASTNodeCpp>, LyukDataTypeCpp>>
LyukParser::parse_arguments_stub(const std::string& raw_args_str) {
    std::vector<std::variant<std::shared_ptr<LyukBasePrimitiveASTNodeCpp>, LyukDataTypeCpp>> parsed_args_vec;
    if (raw_args_str.empty()) return parsed_args_vec;

    std::istringstream args_stream(raw_args_str);
    std::string token;
    
    // This stub just splits by space and tries to guess type for simple literals.
    // It doesn't handle nested structures, quotes properly, operators, etc.
    while (args_stream >> token) {
        if (token.empty()) continue;
        LyukDataTypeCpp arg_data;
        arg_data.is_literal_flag = true;

        if (token == "true" || token == "TRUE") {
            arg_data.type_name_str = "Boolean"; arg_data.value_variant = true;
        } else if (token == "false" || token == "FALSE") {
            arg_data.type_name_str = "Boolean"; arg_data.value_variant = false;
        } else if (token.front() == '"' && token.back() == '"' && token.length() >= 2) {
            arg_data.type_name_str = "String"; arg_data.value_variant = token.substr(1, token.length() - 2);
        } else {
            // Try to parse as number (double then long long)
            try {
                size_t processed_chars_double = 0;
                double d_val = std::stod(token, &processed_chars_double);
                if (processed_chars_double == token.length()) { // Entire token was a double
                    arg_data.type_name_str = "Number"; arg_data.value_variant = d_val;
                    parsed_args_vec.push_back(arg_data); continue;
                }
            } catch (const std::exception&) { /* Not a double or only partially */ }
            try {
                size_t processed_chars_long = 0;
                long long ll_val = std::stoll(token, &processed_chars_long);
                if (processed_chars_long == token.length()) { // Entire token was a long long
                    arg_data.type_name_str = "Number"; arg_data.value_variant = ll_val;
                    parsed_args_vec.push_back(arg_data); continue;
                }
            } catch (const std::exception&) { /* Not a long long */ }
            
            // Default to Symbol if not a recognized literal
            arg_data.type_name_str = "Symbol"; arg_data.value_variant = token;
        }
        parsed_args_vec.push_back(arg_data);
    }
    return parsed_args_vec;
}

// --- Semantic Interpretation (LMI Phase 2 - Stub) ---
std::map<std::string, std::any> LyukParser::interpret_ast_semantically_stub(
    const std::vector<std::shared_ptr<LyukBasePrimitiveASTNodeCpp>>& ast_sequence) {
    
    std::map<std::string, std::any> interpretation_summary_map;
    if (!core_recombinator_) {
        interpretation_summary_map["error_str"] = std::string("Core not available for semantic interpretation.");
        return interpretation_summary_map;
    }
    core_recombinator_->log_message("DEBUG", "LyukParserCpp", 
        "STUB: Performing semantic interpretation on AST with " + std::to_string(ast_sequence.size()) + " root nodes.");

    int let_count = 0;
    int felt_count = 0;
    std::vector<std::string> variable_names_declared_sim_vec;

    for (const auto& node_ptr : ast_sequence) {
        if (!node_ptr) continue;
        if (node_ptr->primitive_name_str == "LET") {
            let_count++;
            if (node_ptr->destination_variable_name_opt_str) {
                variable_names_declared_sim_vec.push_back(*node_ptr->destination_variable_name_opt_str);
            }
        } else if (node_ptr->primitive_name_str == "FELT") {
            felt_count++;
        }
        // A real semantic analyzer would:
        // - Build and manage symbol tables / scopes.
        // - Perform type checking on arguments and expressions.
        // - Resolve symbol references.
        // - Potentially build a control flow graph or intermediate representation.
        // - Infer intentions, emotions, logical assertions from the AST.
    }
    interpretation_summary_map["num_let_statements_sim_val"] = let_count;
    interpretation_summary_map["num_felt_statements_sim_val"] = felt_count;
    // interpretation_summary_map["declared_variables_sample_vec_str"] = variable_names_declared_sim_vec; // std::any can't take vector directly easily for Pybind

    return interpretation_summary_map;
}

// --- Transcompilation (LTC - Stub) ---
std::string LyukParser::transcompile_ast_to_human_readable_stub(
    const std::vector<std::shared_ptr<LyukBasePrimitiveASTNodeCpp>>& ast_sequence,
    const std::string& target_representation_type) {
    
    std::ostringstream oss;
    oss << "// Lyuk AST Transcompiled to: " << target_representation_type << " (C++ Stub)\n";
    oss << "// Total top-level primitives: " << ast_sequence.size() << "\n---\n";

    std::function<void(const std::shared_ptr<LyukBasePrimitiveASTNodeCpp>&, int)> transpile_node_recursive;
    transpile_node_recursive = 
        [&](const std::shared_ptr<LyukBasePrimitiveASTNodeCpp>& node_ptr, int indent_level) {
        if (!node_ptr) return;
        std::string indent_str(indent_level * 2, ' '); // 2 spaces per indent level

        oss << indent_str << "/* Line " << node_ptr->source_line_ref_num << " */ ";
        oss << node_ptr->primitive_name_str;

        if (!node_ptr->parsed_arguments_vec.empty()) {
            oss << "(";
            for (size_t i = 0; i < node_ptr->parsed_arguments_vec.size(); ++i) {
                const auto& arg_variant = node_ptr->parsed_arguments_vec[i];
                if (std::holds_alternative<LyukDataTypeCpp>(arg_variant)) {
                    const auto& data_val = std::get<LyukDataTypeCpp>(arg_variant);
                    oss << mugen_param_value_to_string(data_val.value_variant); // Use helper (if it matches LyukValueVariantCpp)
                                                                              // Or a dedicated to_string for LyukValueVariantCpp
                } else if (std::holds_alternative<std::shared_ptr<LyukBasePrimitiveASTNodeCpp>>(arg_variant)) {
                    oss << "<NestedAST:" << std::get<std::shared_ptr<LyukBasePrimitiveASTNodeCpp>>(arg_variant)->primitive_name_str << ">";
                }
                if (i < node_ptr->parsed_arguments_vec.size() - 1) oss << ", ";
            }
            oss << ")";
        }
        
        if (node_ptr->destination_variable_name_opt_str) {
            oss << " => " << *node_ptr->destination_variable_name_opt_str;
        }
        oss << ";\n";

        // If it's a block-like node (e.g., FRAME, IF, LOOP), recursively transpile its body
        if (auto frame_node = std::dynamic_pointer_cast<LyukFrameASTNodeCpp>(node_ptr)) {
            oss << indent_str << "{\n";
            for (const auto& body_node_ptr : frame_node->body_primitives_vec) {
                transpile_node_recursive(body_node_ptr, indent_level + 1);
            }
            oss << indent_str << "}\n";
        }
        // ... Add similar for IF, LOOP bodies ...
    };

    for (const auto& root_node_ptr : ast_sequence) {
        transpile_node_recursive(root_node_ptr, 0);
    }
    oss << "---\n// End of Transcompilation Stub\n";
    return oss.str();
}
```

```cpp
// eane_cpp_modules/consciousness_module_cpp/consciousness_module.h
#pragma once
#include "../core_interface.h"
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <deque>    // For phi_history_short_term_
#include <map>      // For active_mental_experiments_
#include <optional> // For optional returns or members

// C++ equivalent of Python's ConsciousState dataclass
struct ConsciousStateCpp {
    Eigen::VectorXd perception_vec; // Renamed
    Eigen::VectorXd decision_vec;   // Renamed
    Eigen::VectorXd narrative_vec;  // Renamed

    ConsciousStateCpp(int p_dim, int d_dim, int n_dim) :
        perception_vec(Eigen::VectorXd::Zero(p_dim > 0 ? p_dim : 0)),
        decision_vec(Eigen::VectorXd::Zero(d_dim > 0 ? d_dim : 0)),
        narrative_vec(Eigen::VectorXd::Zero(n_dim > 0 ? n_dim : 0)) {}
};

// C++ equivalent of Python's MentalExperimentLog dataclass (simplified)
struct MentalExperimentLogCpp {
    std::string experiment_id_str;
    std::string creator_query_str;
    double start_timestamp_val = 0.0;
    std::string status_str = "initiated_cpp_exp";
    std::vector<std::string> involved_modules_conceptual_vec;
    // std::vector<std::map<std::string, std::any>> key_insights_generated_vec_sim; // std::any is complex
    std::optional<double> impact_on_phi_observed_opt_val;
    std::optional<std::string> shimyureshon_reflexion_id_opt_str;
    // ... more fields as needed, matching Python version where feasible ...
};


class ConsciousnessModuleCpp {
public:
    ConsciousnessModuleCpp(CoreInterface* core,
                           int perception_dim = 10, 
                           int decision_dim = 3, 
                           int narrative_dim = 5,
                           double phi_modulation_factor = 0.15);
    
    void update_logic();

    // Method to be called by Core when Shimyureshon (for mental experiment) results are ready
    // report_sim: A map representing the ShimyureshonMetricsReport (simplified for C++)
    void process_shimyureshon_reflexion_results_stub(const std::string& sh_id, 
                                                     const std::map<std::string, std::any>& report_sim);

private:
    CoreInterface* core_recombinator_; // Non-owning
    int perception_dim_;
    int decision_dim_;
    int narrative_dim_;
    double phi_modulation_factor_;

    ConsciousStateCpp internal_conscious_state_obj_; // Renamed
    Eigen::MatrixXd W_n_matrix_; // Narrative weights (narrative_dim x (perception_dim + decision_dim))
    Eigen::MatrixXd W_util_matrix_; // Utility/decision weights (perception_dim x decision_dim)

    std::deque<double> phi_history_short_term_deque_; // Maxlen managed manually or by deque if limited
    const size_t phi_history_maxlen_ = 20;

    std::map<std::string, MentalExperimentLogCpp> active_mental_experiments_map_;

    // --- Internal Helper Methods ---
    Eigen::VectorXd make_internal_decision_cm_internal(const Eigen::VectorXd& current_perception_vec); // Renamed
    Eigen::VectorXd build_internal_narrative_cm_internal(const Eigen::VectorXd& current_perception_vec, 
                                                      const Eigen::VectorXd& current_decision_vec); // Renamed
    double compute_phi_detailed_cm_stub() const; // Renamed with _stub

    // Methods for mental experiments (stubs)
    void process_creator_mental_experiment_request_stub();
    void launch_shimyureshon_for_mental_experiment_stub(const std::string& experiment_id, const std::string& query, const std::string& profile_key);
};
```

```cpp
// eane_cpp_modules/consciousness_module_cpp/consciousness_module.cpp
// (Continuation from Part 7)
#include "consciousness_module.h" // Already included

void ConsciousnessModuleCpp::process_shimyureshon_reflexion_results_stub(
    const std::string& sh_id, 
    const std::map<std::string, std::any>& report_sim) {
    
    if (!core_recombinator_) return;
    core_recombinator_->log_message("INFO", "ConsciousnessModuleCpp", "STUB: Processing Shimyureshon results for SH_ID: " + sh_id);

    std::string experiment_id_found;
    auto it_exp = std::find_if(active_mental_experiments_map_.begin(), active_mental_experiments_map_.end(),
        [&sh_id](const auto& pair_exp) {
            return pair_exp.second.shimyureshon_reflexion_id_opt_str.has_value() &&
                   *(pair_exp.second.shimyureshon_reflexion_id_opt_str) == sh_id;
        });

    if (it_exp == active_mental_experiments_map_.end()) {
        core_recombinator_->log_message("WARNING", "ConsciousnessModuleCpp", "No active mental experiment found for Shimyureshon ID: " + sh_id);
        return;
    }
    MentalExperimentLogCpp& log_entry = it_exp->second;
    experiment_id_found = it_exp->first;

    // STUB: Extract relevant metrics from report_sim (which is std::any map)
    // This requires careful casting.
    double sh_success_score_sim = 0.7; // Placeholder
    std::string sh_summary_insight_sim = "Shimyureshon provided conceptual insights.";
    // if (report_sim.count("custom_scenario_metrics_map_any")) { ... cast and extract ... }

    log_entry.status_str = "reflection_generated_from_sh_cpp";
    // log_entry.resolution_summary_str = sh_summary_insight_sim;
    // log_entry.understanding_depth_score_calc_val = sh_success_score_sim * 0.8; // Conceptual
    log_entry.impact_on_phi_observed_opt_val = core_recombinator_->get_global_state_const().phi_consciousness;
    // log_entry.completion_timestamp_val = core_recombinator_->get_current_timestamp();

    core_recombinator_->log_message("INFO", "ConsciousnessModuleCpp", 
        "Mental experiment '" + log_entry.experiment_id_str + "' (from SH: " + sh_id + ") processing completed (stub).");

    // Send event to NarrativeSelfCpp (conceptual)
    // EventDataCpp ns_event("mental_experiment_resolution_fused_for_ns_cpp");
    // ... populate content ...
    // core_recombinator_->event_queue_put(ns_event);

    active_mental_experiments_map_.erase(experiment_id_found); // Remove completed experiment
}


// STUB for launching Shimyureshon
void ConsciousnessModuleCpp::launch_shimyureshon_for_mental_experiment_stub(
    const std::string& experiment_id, const std::string& query, const std::string& profile_key) {
    if (!core_recombinator_) return;
    core_recombinator_->log_message("INFO", "ConsciousnessModuleCpp", 
        "STUB: Requesting Shimyureshon for experiment ID: " + experiment_id + ", Profile: " + profile_key);
    
    if (active_mental_experiments_map_.count(experiment_id)) {
        active_mental_experiments_map_[experiment_id].shimyureshon_reflexion_id_opt_str = "sh_sim_id_" + experiment_id;
        active_mental_experiments_map_[experiment_id].status_str = "shimyureshon_running_cpp_sim";
    }
    // In a real system, this would prepare parameters and call core_recombinator_->start_shimyureshon_cpp(...)
    // For the stub, we can simulate that it starts and will eventually complete.
    // A callback or polling mechanism would be needed for results.
}

// STUB for processing mental experiment requests
void ConsciousnessModuleCpp::process_creator_mental_experiment_request_stub() {
    if(!core_recombinator_) return;
    // Conceptual: Check event queue for "creator_mental_experiment_input_cpp"
    // EventDataCpp exp_request_event;
    // bool found = core_recombinator_->event_queue_get_specific_cpp(
    //                  "creator_mental_experiment_input_cpp", exp_request_event, 0.001);
    // if (found) {
    //     std::string exp_id = "mexp_cpp_default";
    //     std::string query_str = "Default C++ query";
    //     std::string profile_key_str = "default_deep_dive_cpp";
    //     if(exp_request_event.content.count("experiment_id_str")) { try{ exp_id = std::get<std::string>(exp_request_event.content.at("experiment_id_str")); } catch(...){} }
    //     if(exp_request_event.content.count("query_str_val")) { try{ query_str = std::get<std::string>(exp_request_event.content.at("query_str_val")); } catch(...){} }
    //     // ... get profile_key ...
    //
    //     if (active_mental_experiments_map_.find(exp_id) == active_mental_experiments_map_.end()) {
    //         MentalExperimentLogCpp new_log;
    //         new_log.experiment_id_str = exp_id;
    //         new_log.creator_query_str = query_str;
    //         new_log.start_timestamp_val = core_recombinator_->get_current_timestamp();
    //         active_mental_experiments_map_[exp_id] = new_log;
    //         launch_shimyureshon_for_mental_experiment_stub(exp_id, query_str, profile_key_str);
    //     }
    // }
}
```

```cpp
// eane_cpp_modules/emotion_regulation_module_cpp/emotion_regulation_module.h
// (Header from Part 8 is largely complete, no major changes needed for this stub)
// Might add rng_erm_ and uniform_dist_lm_01_ for disturbance simulation if not getting from Python
#pragma once
#include "../core_interface.h"
#include <random> // For simulating disturbance

struct EmotionStateDataCpp { // Already defined in Part 8
    double valence;
    double arousal;
};

class EmotionRegulationModuleCpp {
public:
    EmotionRegulationModuleCpp(CoreInterface* core,
                               double reference_valence = 0.15, double reference_arousal = 0.4,
                               double kp = 0.3, double ki = 0.06, double kd = 0.03,
                               double dt_factor_erm = 1.0); // dt_factor from Python
    void update_logic();

private:
    CoreInterface* core_recombinator_;
    EmotionStateDataCpp reference_state_erm_;
    double kp_erm_, ki_erm_, kd_erm_;
    double dt_factor_erm_; // To align with Python version's time scaling
    EmotionStateDataCpp integral_error_erm_ = {0.0, 0.0};
    EmotionStateDataCpp previous_error_erm_ = {0.0, 0.0};

    EmotionStateDataCpp compute_current_error_erm(const EmotionStateDataCpp& current_state_in) const; // Made const
    EmotionStateDataCpp pid_control_signal_erm(const EmotionStateDataCpp& error_in);
    
    // For simulating disturbance if not received via event
    mutable std::mt19937 rng_erm_internal_;
    mutable std::normal_distribution<double> disturbance_dist_erm_;
};
```

```cpp
// eane_cpp_modules/emotion_regulation_module_cpp/emotion_regulation_module.cpp
// (Header from Part 8 is largely complete, implementation was also there)
// (Small refinement to use internal RNG for disturbance if no event)
#include "emotion_regulation_module.h"
#include <algorithm> // std::clamp

EmotionRegulationModuleCpp::EmotionRegulationModuleCpp(CoreInterface* core,
                                                   double reference_valence, double reference_arousal,
                                                   double kp, double ki, double kd, double dt_factor_erm_in) // Added dt_factor
    : core_recombinator_(core), reference_state_erm_({reference_valence, reference_arousal}),
      kp_erm_(kp), ki_erm_(ki), kd_erm_(kd), dt_factor_erm_(dt_factor_erm_in),
      rng_erm_internal_(std::random_device{}()), disturbance_dist_erm_(0.0, 0.005) { // Small disturbance
    if(core_recombinator_) core_recombinator_->log_message("INFO", "EmotionRegulationCpp", "EmotionRegulationModule C++ (ERM) initialized.");
}

void EmotionRegulationModuleCpp::update_logic() {
    if(!core_recombinator_) return;
    GlobalSelfStateCpp& gs = core_recombinator_->get_global_state();
    
    EmotionStateDataCpp current_emotion_gs = {gs.valencia, gs.arousal};
    EmotionStateDataCpp disturbance_val = {disturbance_dist_erm_(rng_erm_internal_), disturbance_dist_erm_(rng_erm_internal_)};

    // STUB: Process event for explicit emotional_perturbation_input_cpp
    // EventDataCpp disturbance_event;
    // if (core_recombinator_->event_queue_get_specific_cpp("emotional_perturbation_input_cpp", disturbance_event, 0.001)) {
    //     if(disturbance_event.content.count("valence_change_val")) {
    //        try{ disturbance_val.valence += std::get<double>(disturbance_event.content.at("valence_change_val")); } catch(...){}
    //     }
    //     // ... similar for arousal_change_val ...
    // }

    EmotionStateDataCpp error_state = compute_current_error_erm(current_emotion_gs);
    EmotionStateDataCpp control_signal = pid_control_signal_erm(error_state);
    
    // Use the dt_factor_erm_ passed from Python's update_interval concept for this module
    double effective_dt_module = gs.time_delta_continuous_step * dt_factor_erm_;

    gs.valencia = std::clamp(gs.valencia + control_signal.valence * effective_dt_module + disturbance_val.valence * effective_dt_module, -1.0, 1.0);
    gs.arousal = std::clamp(gs.arousal + control_signal.arousal * effective_dt_module + disturbance_val.arousal * effective_dt_module, 0.05, 1.0);
}

EmotionStateDataCpp EmotionRegulationModuleCpp::compute_current_error_erm(const EmotionStateDataCpp& current_state_in) const {
    return {reference_state_erm_.valence - current_state_in.valence,
            reference_state_erm_.arousal - current_state_in.arousal};
}

EmotionStateDataCpp EmotionRegulationModuleCpp::pid_control_signal_erm(const EmotionStateDataCpp& error_in) {
    if(!core_recombinator_) return {0.0, 0.0};
    // dt_factor_erm_ is critical here for PID tuning to match Python behavior
    double effective_dt_pid = core_recombinator_->get_time_delta_continuous() * dt_factor_erm_;
    if (effective_dt_pid < 1e-9) effective_dt_pid = 0.01; // Avoid division by zero

    integral_error_erm_.valence = std::clamp(integral_error_erm_.valence + error_in.valence * effective_dt_pid, -2.0 / ki_erm_ , 2.0 / ki_erm_); // Anti-windup
    integral_error_erm_.arousal = std::clamp(integral_error_erm_.arousal + error_in.arousal * effective_dt_pid, -2.0 / ki_erm_, 2.0 / ki_erm_);

    double derivative_valence = (error_in.valence - previous_error_erm_.valence) / effective_dt_pid;
    double derivative_arousal = (error_in.arousal - previous_error_erm_.arousal) / effective_dt_pid;

    double control_v = kp_erm_ * error_in.valence + ki_erm_ * integral_error_erm_.valence + kd_erm_ * derivative_valence;
    double control_a = kp_erm_ * error_in.arousal + ki_erm_ * integral_error_erm_.arousal + kd_erm_ * derivative_arousal;

    previous_error_erm_ = error_in;
    return {control_v, control_a};
}
```

The headers for `llyuk_communication_module_cpp`, `social_dynamics_module_cpp`, `ontology_flow_manager_cpp`, `needs_manager_cpp`, `craving_module_cpp`, `self_compassion_module_cpp`, `stress_response_module_cpp`, `pain_matrix_directive_cpp`, `defense_mechanisms_cpp` from previous parts are assumed to be complete as stubs. Their `.cpp` implementations would be simple initializations and placeholder `update_logic` methods if not already detailed.

```cpp
// eane_cpp_modules/pybind_wrapper.cpp
// (Final Version - Combining all previous parts and ensuring all modules are registered)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>            // For std::vector, std::map, std::string, std::optional
#include <pybind11/eigen.h>          // For Eigen::VectorXd, Eigen::MatrixXd
#include <pybind11/functional.h>     // For std::function
#include <pybind11/chrono.h>         // For std::chrono types if used in interfaces
#include <pybind11/iostream.h>       // For py::add_ostream_redirect
#include <pybind11/operators.h>      // For custom operators if needed
// #include <pybind11/stl_bind.h>    // For binding opaque types like std::deque (if directly exposed)
// PYBIND11_MAKE_OPAQUE(std::deque<double>); // Example for deque

// Core Interface
#include "core_interface.h"

// Mathematical Toolkit & Simulators
#include "mathematical_toolkit/mathematical_toolkit.h"
#include "physics_simulators/physics_simulators.h"

// Foundational Cognitive Modules
#include "subconscious_mind/subconscious_mind.h"
#include "learning_module/learning_module.h" // Includes its stubs
#include "self_evolution_module/self_evolution_module.h" // Includes sem_types.h
#include "freewill_module/freewill_module.h"       // Includes freewill_types.h
#include "freewill_engine/freewill_engine.h"       // Includes environment_fwe.h

// Interface & Support Modules
#include "adaptive_firewall_module/adaptive_firewall_module.h" // Includes firewall_types.h
#include "timeseries_predictor_module/timeseries_predictor_module.h" // Includes ts_types.h

// Advanced Processing & Lyuk
#include "controlled_mutation_generator/controlled_mutation_generator.h" // Includes mugen_types.h
#include "lyuk_parser/lyuk_parser.h"                         // Includes lyuk_ast_types.h

// Homeostasis & Regulation Modules (Stubs/Implementations)
#include "consciousness_module_cpp/consciousness_module.h"
#include "emotion_regulation_module_cpp/emotion_regulation_module.h"
#include "needs_manager_cpp/needs_manager.h"
#include "craving_module_cpp/craving_module.h"
#include "self_compassion_module_cpp/self_compassion_module.h"
#include "stress_response_module_cpp/stress_response_module.h"
#include "pain_matrix_directive_cpp/pain_matrix_directive.h"
#include "defense_mechanisms_cpp/defense_mechanisms.h"

// Communication & Social Modules (Stubs)
#include "llyuk_communication_module_cpp/llyuk_communication_module.h"
#include "social_dynamics_module_cpp/social_dynamics_module.h"
#include "ontology_flow_manager_cpp/ontology_flow_manager.h"


namespace py = pybind11;
using namespace pybind11::literals; // For _a (named arguments)

// --- PyCoreInterface (Implementation of CoreInterface for C++ to call Python Core) ---
// (This implementation was provided in Part 1 and refined. Ensure it's complete here.)
class PyCoreInterface : public CoreInterface {
public:
    // Store the Python CoreRecombinator object
    py::object py_core_recombinator_obj_; // Renamed for clarity
    mutable GlobalSelfStateCpp gs_cache_; // Cache for GS, mutable for const getter to log if out of sync

    // Constructor that takes the Python CoreRecombinator object
    explicit PyCoreInterface(py::object py_core_obj) : py_core_recombinator_obj_(std::move(py_core_obj)) {
        py::gil_scoped_acquire gil;
        if (py_core_recombinator_obj_.is_none()) {
            std::cerr << "PyCoreInterface C++ Warning: Python CoreRecombinator object is None during construction." << std::endl;
            // Initialize gs_cache_ with defaults if no Python core
            gs_cache_ = GlobalSelfStateCpp();
        } else {
             log_message("DEBUG", "PyCoreInterface", "PyCoreInterface constructed with Python CoreRecombinator object.");
            // Initial sync of GlobalSelfState
            if (py::hasattr(py_core_recombinator_obj_, "global_state")) {
                py::object py_gs = py_core_recombinator_obj_.attr("global_state");
                if (!py_gs.is_none()) {
                    sync_gs_from_python_internal(py_gs);
                } else {
                     log_message("WARNING", "PyCoreInterface", "Python global_state is None during initial sync.");
                }
            } else {
                 log_message("WARNING", "PyCoreInterface", "Python CoreRecombinator has no global_state attribute during initial sync.");
            }
        }
    }
    // Default constructor for C++-only tests (no Python core)
    PyCoreInterface() : py_core_recombinator_obj_(py::none()) {
        // std::cout << "PyCoreInterface C++: Default constructed (no Python core)." << std::endl;
        gs_cache_ = GlobalSelfStateCpp(); // Initialize with defaults
    }

    // --- CoreInterface Implementation ---
    GlobalSelfStateCpp& get_global_state() override {
        py::gil_scoped_acquire gil;
        if (!py_core_recombinator_obj_.is_none() && py::hasattr(py_core_recombinator_obj_, "global_state")) {
            try {
                py::object py_gs = py_core_recombinator_obj_.attr("global_state");
                if (!py_gs.is_none()) {
                    sync_gs_from_python_internal(py_gs);
                } else { /* Log warning handled by sync_gs_from_python_internal */ }
            } catch (const py::error_already_set& e) {
                log_message("ERROR", "PyCoreInterface", std::string("get_global_state: Python error syncing GS: ") + e.what());
            }
        }
        return gs_cache_;
    }

    const GlobalSelfStateCpp& get_global_state_const() const override {
        // For const version, ideally, we'd ensure gs_cache_ is up-to-date without modifying it here.
        // This could involve a periodic sync or a flag. For simplicity, it returns current cache.
        // A log message could indicate potential staleness if not recently synced.
        // py::gil_scoped_acquire gil; // Not strictly needed if only reading gs_cache_
        // log_message("DEBUG", "PyCoreInterface", "get_global_state_const returning cached GS. Ensure recent sync for accuracy.");
        return gs_cache_;
    }

    void event_queue_put(const EventDataCpp& event_data) override {
        py::gil_scoped_acquire gil;
        if (py_core_recombinator_obj_.is_none() || !py::hasattr(py_core_recombinator_obj_, "event_queue_put")) {
            log_message("ERROR", "PyCoreInterface", "event_queue_put: No Python core or event_queue_put method. Event Type: " + event_data.type);
            return;
        }
        try {
            py::dict py_event_dict;
            py_event_dict["type"] = event_data.type;
            py::dict py_content_dict;
            for (const auto& pair_item : event_data.content) {
                std::visit([&](auto&& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (!std::is_same_v<T, std::monostate>) {
                        // pybind11 handles conversion for most types in the variant
                        py_content_dict[pair_item.first.c_str()] = py::cast(arg);
                    }
                }, pair_item.second);
            }
            py_event_dict["content"] = py_content_dict;
            if (event_data.source_module) py_event_dict["source_module"] = *event_data.source_module;
            if (event_data.target_module) py_event_dict["target_module_hint_cpp"] = *event_data.target_module; // Hint for Python routing

            py_core_recombinator_obj_.attr("event_queue_put")(py_event_dict, "priority_label"_a = event_data.priority_label);
        } catch (const py::error_already_set& e) {
            log_message("ERROR", "PyCoreInterface", std::string("event_queue_put: Python error: ") + e.what() + " for event type " + event_data.type);
        } catch (const std::exception& e) { // Catch other C++ exceptions during conversion
            log_message("ERROR", "PyCoreInterface", std::string("event_queue_put: C++ std::exception: ") + e.what());
        }
    }

    double get_current_timestamp() const override {
        py::gil_scoped_acquire gil;
        if (!py_core_recombinator_obj_.is_none() && py::hasattr(py_core_recombinator_obj_, "global_state") &&
            !py_core_recombinator_obj_.attr("global_state").is_none() && py::hasattr(py_core_recombinator_obj_.attr("global_state"), "timestamp")) {
            try {
                return py_core_recombinator_obj_.attr("global_state").attr("timestamp").cast<double>();
            } catch (const py::error_already_set& e) {
                 log_message("ERROR", "PyCoreInterface", std::string("get_current_timestamp: Python error: ") + e.what());
            }
        }
        return gs_cache_.system_timestamp; // Fallback to cached
    }

    int get_current_cycle_num() const override {
        py::gil_scoped_acquire gil;
        if (!py_core_recombinator_obj_.is_none() && py::hasattr(py_core_recombinator_obj_, "current_cycle_num")) {
             try {
                return py_core_recombinator_obj_.attr("current_cycle_num").cast<int>();
            } catch (const py::error_already_set& e) {
                log_message("ERROR", "PyCoreInterface", std::string("get_current_cycle_num: Python error: ") + e.what());
            }
        }
        return -1; // Fallback
    }
    double get_time_delta_continuous() const override {
        py::gil_scoped_acquire gil;
        if (!py_core_recombinator_obj_.is_none() && py::hasattr(py_core_recombinator_obj_, "global_state") &&
            !py_core_recombinator_obj_.attr("global_state").is_none() && py::hasattr(py_core_recombinator_obj_.attr("global_state"), "time_delta_continuous")) {
            try {
                return py_core_recombinator_obj_.attr("global_state").attr("time_delta_continuous").cast<double>();
            } catch (const py::error_already_set& e) {
                log_message("ERROR", "PyCoreInterface", std::string("get_time_delta_continuous: Python error: ") + e.what());
            }
        }
        return gs_cache_.time_delta_continuous_step; // Fallback
    }

    void log_message(const std::string& level, const std::string& module_name_cpp, const std::string& message) const override {
        // This const_cast is generally okay if Python logging is thread-safe or GIL protects.
        // A better way for truly const logging might involve a thread-safe queue.
        PyCoreInterface* non_const_this = const_cast<PyCoreInterface*>(this);
        if (non_const_this->py_core_recombinator_obj_.is_none()) {
            std::cout << "[" << level << "] C++ " << module_name_cpp << " (no_core_log): " << message << std::endl;
            return;
        }
        py::gil_scoped_acquire gil;
        try {
            // Prefer using a dedicated logger from the Python core if available
            if (py::hasattr(non_const_this->py_core_recombinator_obj_, "get_logger_for_module_cpp")) {
                py::object logger = non_const_this->py_core_recombinator_obj_.attr("get_logger_for_module_cpp")(module_name_cpp);
                std::string upper_level = level;
                std::transform(upper_level.begin(), upper_level.end(), upper_level.begin(), ::toupper);

                if (upper_level == "DEBUG") logger.attr("debug")(message);
                else if (upper_level == "INFO") logger.attr("info")(message);
                else if (upper_level == "WARNING") logger.attr("warning")(message);
                else if (upper_level == "ERROR") logger.attr("error")(message);
                else if (upper_level == "CRITICAL") logger.attr("critical")(message);
                else logger.attr("info")(std::string("[") + level + "] " + message); // Default to info
            } else if (py::hasattr(non_const_this->py_core_recombinator_obj_, "logger")) { // Fallback to a generic core logger
                 non_const_this->py_core_recombinator_obj_.attr("logger").attr("info")(
                    std::string("C++ ") + module_name_cpp + " [" + level + "]: " + message);
            } else { // Absolute fallback
                 std::cout << "[" << level << "] C++ " << module_name_cpp << " (py_cout_log): " << message << std::endl;
            }
        } catch (const py::error_already_set& e) {
            std::cerr << "PyCoreInterface::log_message Python error: " << e.what() << std::endl;
            std::cout << "[" << level << "] C++ " << module_name_cpp << " (py_exception_fallback_log): " << message << std::endl;
        }
    }

private:
    // Helper to sync gs_cache_ from Python global_state object
    void sync_gs_from_python_internal(py::object py_gs) {
        // This function assumes GIL is held by caller.
        if (py_gs.is_none()) {
            log_message("WARNING", "PyCoreInterface", "sync_gs_from_python_internal: Python global_state (py_gs) is None.");
            return;
        }
        try {
            // Macro to simplify attribute getting and casting with error handling
            #define GET_GS_ATTR(py_obj, attr_name, cpp_member, cpp_type, default_val) \
                if (py::hasattr(py_obj, #attr_name)) { \
                    try { cpp_member = py_obj.attr(#attr_name).cast<cpp_type>(); } \
                    catch (const py::cast_error& e) { \
                        log_message("WARNING", "PyCoreInterface", std::string("Cast error for GS attr '") + #attr_name + "': " + e.what()); \
                        cpp_member = default_val; \
                    } \
                } else { cpp_member = default_val; log_message("WARNING", "PyCoreInterface", std::string("GS attr '") + #attr_name + "' not found in Python object."); }

            GET_GS_ATTR(py_gs, valencia, gs_cache_.valencia, double, 0.0);
            GET_GS_ATTR(py_gs, arousal, gs_cache_.arousal, double, 0.5);
            GET_GS_ATTR(py_gs, motivacion, gs_cache_.motivacion, double, 0.5);
            GET_GS_ATTR(py_gs, dolor, gs_cache_.dolor, double, 0.0);

            if (py::hasattr(py_gs, "needs") && py_gs.attr("needs").is_instance<py::array_t<double>>()) {
                py::array_t<double> py_needs_arr = py_gs.attr("needs").cast<py::array_t<double>>();
                if (py_needs_arr.ndim() == 1 && py_needs_arr.size() == 3) {
                    gs_cache_.needs_vector = Eigen::Map<Eigen::Vector3d>(py_needs_arr.mutable_data());
                } else {
                    log_message("WARNING", "PyCoreInterface", "GS attr 'needs' is not a 1D array of size 3. Using default.");
                    gs_cache_.needs_vector << 0.7, 0.7, 0.7;
                }
            } else { gs_cache_.needs_vector << 0.7, 0.7, 0.7; }
            
            // Beliefs distribution (example if it's a NumPy array)
            if (py::hasattr(py_gs, "beliefs") && py_gs.attr("beliefs").is_instance<py::array_t<double>>()) {
                py::array_t<double> py_beliefs_arr = py_gs.attr("beliefs").cast<py::array_t<double>>();
                if (py_beliefs_arr.ndim() == 1) { // Assuming 1D for simplicity
                    gs_cache_.beliefs_distribution.resize(py_beliefs_arr.size());
                    for(py::ssize_t i=0; i < py_beliefs_arr.size(); ++i) gs_cache_.beliefs_distribution(i) = py_beliefs_arr.at(i);
                } else { gs_cache_.beliefs_distribution = Eigen::Vector3d(1.0/3.0, 1.0/3.0, 1.0/3.0); }
            } else { gs_cache_.beliefs_distribution = Eigen::Vector3d(1.0/3.0, 1.0/3.0, 1.0/3.0); }


            GET_GS_ATTR(py_gs, phi_consciousness, gs_cache_.phi_consciousness, double, 0.0);
            GET_GS_ATTR(py_gs, phi_funcional_score, gs_cache_.phi_funcional_score, double, 0.0);
            GET_GS_ATTR(py_gs, coherence_score, gs_cache_.coherence_score, double, 0.75);
            GET_GS_ATTR(py_gs, synchrony, gs_cache_.synchrony_metric, double, 0.7); // synchrony -> synchrony_metric
            GET_GS_ATTR(py_gs, system_entropy, gs_cache_.system_entropy, double, 0.12);
            GET_GS_ATTR(py_gs, self_esteem, gs_cache_.self_esteem, double, 0.7);
            GET_GS_ATTR(py_gs, qualia_state, gs_cache_.qualia_state_label, std::string, "neutral_adaptativo_cpp");
            
            // Values map
            if (py::hasattr(py_gs, "values") && py_gs.attr("values").is_instance<py::dict>()) {
                py::dict py_values_dict = py_gs.attr("values").cast<py::dict>();
                gs_cache_.values.clear();
                for (auto item : py_values_dict) {
                    try {
                        gs_cache_.values[item.first.cast<std::string>()] = item.second.cast<double>();
                    } catch (const py::cast_error& e) {
                        log_message("WARNING", "PyCoreInterface", std::string("Cast error for GS values map item: ") + e.what());
                    }
                }
            } else { /* gs_cache_.values retains its default or previous values */ }
            
            GET_GS_ATTR(py_gs, system_id, gs_cache_.system_id_tag, std::string, "EANE_Unknown_Cpp");
            GET_GS_ATTR(py_gs, timestamp, gs_cache_.system_timestamp, double, 0.0);
            GET_GS_ATTR(py_gs, time_delta_continuous, gs_cache_.time_delta_continuous_step, double, 0.1);
            GET_GS_ATTR(py_gs, system_threat_level, gs_cache_.system_threat_level_value, double, 0.05);
            GET_GS_ATTR(py_gs, resilience_stability, gs_cache_.resilience_stability_metric, double, 0.9);
            GET_GS_ATTR(py_gs, circadian_activity_level, gs_cache_.circadian_activity_level_value, double, 0.6);

            // Complex types like goals, meta_actual, decision, current_focus (std::map<string, std::any>)
            // require careful, type-aware conversion. This is a major source of complexity.
            // For this stub, we might skip them or only sync very simple representations.
            // sync_complex_map_from_py(py_gs, "goals", gs_cache_.goals_map_cpp);
            // ...

            #undef GET_GS_ATTR

        } catch (const py::error_already_set& e) { // Catch Python errors during attribute access
            log_message("ERROR", "PyCoreInterface", std::string("sync_gs_from_python_internal: Python error: ") + e.what());
        } catch (const std::exception& e) { // Catch C++ exceptions (e.g. bad_any_cast from our side)
             log_message("ERROR", "PyCoreInterface", std::string("sync_gs_from_python_internal: C++ std::exception: ") + e.what());
        }
    }
    // void sync_complex_map_from_py(py::object py_parent, const char* attr_name, std::map<std::string, std::any>& cpp_target_map); // Example helper
};


PYBIND11_MODULE(eane_cpp_core, m) {
    m.doc() = "EANE C++ Core Modules for Phoenix EANE_6.0 V16.0 - Final Combined";
    py::add_ostream_redirect(m, "ostream_redirect"); // Redirect C++ std::cout/cerr to Python

    // --- Core Utilities Bindings ---
    // GlobalSelfStateCpp
    py::class_<GlobalSelfStateCpp>(m, "GlobalSelfStateCpp")
        .def(py::init<>())
        // Expose members that are simple types or Eigen types (pybind11/eigen.h handles Eigen)
        .def_readwrite("valencia", &GlobalSelfStateCpp::valencia)
        .def_readwrite("arousal", &GlobalSelfStateCpp::arousal)
        .def_readwrite("motivacion", &GlobalSelfStateCpp::motivacion)
        .def_readwrite("dolor", &GlobalSelfStateCpp::dolor)
        .def_readwrite("needs_vector", &GlobalSelfStateCpp::needs_vector)
        .def_readwrite("beliefs_distribution", &GlobalSelfStateCpp::beliefs_distribution)
        .def_readwrite("phi_consciousness", &GlobalSelfStateCpp::phi_consciousness)
        .def_readwrite("phi_funcional_score", &GlobalSelfStateCpp::phi_funcional_score)
        .def_readwrite("coherence_score", &GlobalSelfStateCpp::coherence_score)
        .def_readwrite("synchrony_metric", &GlobalSelfStateCpp::synchrony_metric)
        .def_readwrite("system_entropy", &GlobalSelfStateCpp::system_entropy)
        .def_readwrite("self_esteem", &GlobalSelfStateCpp::self_esteem)
        .def_readwrite("qualia_state_label", &GlobalSelfStateCpp::qualia_state_label)
        .def_readwrite("values", &GlobalSelfStateCpp::values) // std::map<string, double>
        // goals_map_cpp, current_top_goal_cpp, etc. are std::map<string, std::any> - harder to bind directly.
        // Python would typically pass dicts, and C++ would parse them.
        .def_readwrite("system_id_tag", &GlobalSelfStateCpp::system_id_tag)
        .def_readwrite("system_timestamp", &GlobalSelfStateCpp::system_timestamp)
        .def_readwrite("time_delta_continuous_step", &GlobalSelfStateCpp::time_delta_continuous_step)
        .def_readwrite("system_threat_level_value", &GlobalSelfStateCpp::system_threat_level_value)
        .def_readwrite("resilience_stability_metric", &GlobalSelfStateCpp::resilience_stability_metric)
        .def_readwrite("circadian_activity_level_value", &GlobalSelfStateCpp::circadian_activity_level_value);
        // Add .def_property for complex map access if needed, e.g., get_goal_value(goal_id, key)

    // EventDataCpp and EventContentValueCpp (std::variant) are tricky for direct binding.
    // C++ modules will construct EventDataCpp and pass to CoreInterface::event_queue_put.
    // Python events will be received by PyCoreInterface and translated if necessary.

    // CoreInterface (exposed as PyCoreInterface)
    py::class_<CoreInterface, PyCoreInterface /* Trampoline class */>(m, "CoreInterfaceCpp")
        .def(py::init<py::object>(), "py_core_recombinator_obj"_a, 
             "Constructor that takes the Python CoreRecombinator object to bridge C++ calls to Python.")
        .def(py::init<>(), "Default constructor for C++-only testing (creates a CoreInterface without Python linkage).")
        .def("get_global_state_cpp_ref", &CoreInterface::get_global_state, py::return_value_policy::reference_internal,
             "Gets a modifiable reference to the C++ mirror of GlobalSelfState (synchronized on call).")
        .def("get_global_state_cpp_const_ref", &CoreInterface::get_global_state_const, py::return_value_policy::reference_internal,
             "Gets a const reference to the C++ mirror of GlobalSelfState (cached, ensure recent sync).")
        // event_queue_put is called by C++ modules, not directly by Python typically.
        .def("get_current_timestamp_cpp", &CoreInterface::get_current_timestamp)
        .def("get_current_cycle_num_cpp", &CoreInterface::get_current_cycle_num)
        .def("get_time_delta_continuous_cpp", &CoreInterface::get_time_delta_continuous)
        .def("log_message_from_py_to_cpp_core_logger", &CoreInterface::log_message, // Python can use this to log via C++ path
            "level_str"_a, "module_name_str"_a, "message_str"_a);


    // --- MathematicalToolkit & Simulators ---
    // (Bindings from Part 1, ensure they are complete and correct)
    py::class_<MathematicalToolkit>(m, "MathematicalToolkitCpp")
        .def(py::init<CoreInterface*>(), "core_interface_ptr"_a.none(true))
        .def("solve_linear_system", &MathematicalToolkit::solve_linear_system)
        .def("eigen_decomposition", &MathematicalToolkit::eigen_decomposition, "Returns Eigen::SelfAdjointEigenSolver object.")
        // Expose eigenvalues and eigenvectors separately for easier Python use
        .def("get_eigenvalues", [](const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>& solver){ return solver.eigenvalues(); })
        .def("get_eigenvectors", [](const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>& solver){ return solver.eigenvectors(); })
        .def("integrate_ode_rk4", &MathematicalToolkit::integrate_ode_rk4)
        .def("numerical_derivative_scalar_stub", &MathematicalToolkit::numerical_derivative_scalar_stub)
        .def("minimize_scalar_function_gradient_descent_stub", &MathematicalToolkit::minimize_scalar_function_gradient_descent_stub)
        .def("calculate_shannon_entropy", &MathematicalToolkit::calculate_shannon_entropy)
        .def("normal_pdf", &MathematicalToolkit::normal_pdf)
        .def("normal_cdf", &MathematicalToolkit::normal_cdf)
        .def("generate_normal_samples", &MathematicalToolkit::generate_normal_samples)
        .def("generate_uniform_samples", &MathematicalToolkit::generate_uniform_samples)
        .def("get_constants", &MathematicalToolkit::get_constants, py::return_value_policy::reference_internal)
        .def("qms", static_cast<QuantumMechanicsSimulator& (MathematicalToolkit::*)()>(&MathematicalToolkit::qms), py::return_value_policy::reference_internal)
        .def("csm", static_cast<CosmologySimulator& (MathematicalToolkit::*)()>(&MathematicalToolkit::csm), py::return_value_policy::reference_internal)
        .def("ssm", static_cast<StochasticSimulator& (MathematicalToolkit::*)()>(&MathematicalToolkit::ssm), py::return_value_policy::reference_internal);

    py::class_<QuantumMechanicsSimulator>(m, "QuantumMechanicsSimulatorCpp")
        .def("solve_schrodinger_1d_finite_diff", &QuantumMechanicsSimulator::solve_schrodinger_1d_finite_diff,
            "potential_func_eV"_a, "x_min_angstrom"_a, "x_max_angstrom"_a, "num_points"_a,
            "num_eigenstates_to_return"_a = 3, "particle_mass_amu"_a = 1.007276);

    py::class_<CosmologySimulator>(m, "CosmologySimulatorCpp")
        .def("scale_factor_lcdm_flat", &CosmologySimulator::scale_factor_lcdm_flat, "t_lookback_gyr_vec"_a)
        .def("hubble_parameter_at_z", &CosmologySimulator::hubble_parameter_at_z, "redshift"_a);

    py::class_<StochasticSimulator>(m, "StochasticSimulatorCpp")
        .def("generate_poisson_process_events", &StochasticSimulator::generate_poisson_process_events, "mean_rate"_a, "time_duration"_a)
        .def("generate_random_walk_1d", &StochasticSimulator::generate_random_walk_1d, "num_steps"_a, "step_size_std_dev"_a = 1.0, "initial_pos"_a = 0.0);

    // --- Foundational Cognitive Modules ---
    // SubconsciousMind (Bindings from Part 4)
    py::class_<SubconsciousMind::SnapshotDataCpp>(m, "SubconsciousMindSnapshotDataCpp")
        .def(py::init<>())
        .def_readwrite("transition_matrix", &SubconsciousMind::SnapshotDataCpp::transition_matrix)
        .def_readwrite("emission_matrix", &SubconsciousMind::SnapshotDataCpp::emission_matrix)
        .def_readwrite("Wh_matrix", &SubconsciousMind::SnapshotDataCpp::Wh_matrix)
        .def_readwrite("hidden_state_vec", &SubconsciousMind::SnapshotDataCpp::hidden_state_vec);

    py::class_<SubconsciousMind>(m, "SubconsciousMindCpp")
        .def(py::init<CoreInterface*, int, int, int>(), "core_interface"_a, 
             "state_dim"_a = 10, "output_dim_for_consciousness"_a = 10, "num_observation_features"_a = 10)
        .def("update_logic_cpp", &SubconsciousMind::update_logic) // Renamed to avoid clash if Python module also named this
        .def("get_current_influence_output_cpp", &SubconsciousMind::get_current_influence_output)
        .def("get_current_influence_norm_cpp", &SubconsciousMind::get_current_influence_norm)
        .def("get_hidden_state_copy_cpp", &SubconsciousMind::get_hidden_state_copy)
        .def("get_snapshot_data_cpp", &SubconsciousMind::get_snapshot_data)
        .def("restore_from_snapshot_data_cpp", &SubconsciousMind::restore_from_snapshot_data, "snapshot_data"_a);
    
    // LearningModule (Bindings from Part 5 & 6, review for completeness)
    // Exposing stubs like LSTMStub, QLearningAgentStub, KnowledgeBaseStub
    py::class_<LSTMStub>(m, "LSTMStubCpp")
        .def(py::init<int, int, int>(), "input_dim"_a, "hidden_dim"_a, "output_dim"_a)
        .def("step", &LSTMStub::step, "input_vec"_a, "prev_state"_a) // prev_state needs binding for LSTMCellStateStub
        .def("process_sequence", &LSTMStub::process_sequence, "input_sequence"_a)
        .def("train_sequence", &LSTMStub::train_sequence, "input_sequence"_a, "target_output_sequence"_a, "learning_rate"_a)
        .def("get_last_loss", &LSTMStub::get_last_loss)
        .def("get_input_dim", &LSTMStub::get_input_dim)  // Added getter
        .def("get_output_dim", &LSTMStub::get_output_dim); // Added getter

    py::class_<LSTMCellStateStub>(m, "LSTMCellStateStubCpp")
        .def(py::init<int>(), "hidden_dim"_a=0)
        .def_readwrite("hidden_state", &LSTMCellStateStub::hidden_state)
        .def_readwrite("cell_state", &LSTMCellStateStub::cell_state);


    py::class_<QLearningAgentStub>(m, "QLearningAgentStubCpp")
        .def(py::init<int, int, double, double, double>(), "num_states"_a, "num_actions"_a, "learning_rate"_a=0.1, "discount_factor"_a=0.9, "epsilon"_a=0.1)
        .def("choose_action", &QLearningAgentStub::choose_action, "state"_a)
        .def("update", &QLearningAgentStub::update, "state"_a, "action"_a, "reward"_a, "next_state"_a)
        .def("get_q_value", &QLearningAgentStub::get_q_value, "state"_a, "action"_a);

    py::class_<KnowledgeBaseStub>(m, "KnowledgeBaseStubCpp")
        .def(py::init<size_t>(), "max_size"_a = 1000)
        // Store method takes std::any, which is hard to call directly from Python.
        // Would need a Python-friendly wrapper or specific store methods.
        // .def("store_simple_string_content", [](KnowledgeBaseStub& self, const std::string& id, const std::string& summary_content){
        //    std::map<std::string, std::any> content_map; content_map["summary_str"] = summary_content;
        //    self.store(id, content_map);
        // })
        .def("retrieve_summary_string", [](const KnowledgeBaseStub& self, const std::string& id) -> std::optional<std::string> {
            auto entry_opt = self.retrieve(id);
            if (entry_opt && entry_opt->content.count("summary_str")) {
                try { return std::any_cast<std::string>(entry_opt->content.at("summary_str")); } catch(const std::bad_any_cast&){}
            }
            return std::nullopt;
        })
        .def("search_by_vector_similarity_stub", &KnowledgeBaseStub::search_by_vector_similarity_stub)
        .def("size", &KnowledgeBaseStub::size)
        .def("clear", &KnowledgeBaseStub::clear);
        
    // LearningModule itself
    py::class_<LearningModule>(m, "LearningModuleCpp")
        .def(py::init<CoreInterface*, int, int, int, int, int, size_t>(), "core_interface"_a,
             "input_dim_lstm_base"_a=10, "hidden_dim_lstm_base"_a=20, "output_dim_lstm_base"_a=5,
             "num_states_q_base"_a=10, "num_actions_q_base"_a=4, "kb_max_size"_a=2000)
        .def("update_logic_cpp", &LearningModule::update_logic)
        .def("initiate_learning_on_topic_cpp", &LearningModule::initiate_learning_on_topic, "topic_query"_a, "source"_a="internal_cpp")
        .def("train_supervised_model_conceptual_cpp", &LearningModule::train_supervised_model_conceptual, "data_X"_a, "data_y"_a, "model_type_str"_a, "params"_a=py::dict())
        .def("train_unsupervised_model_conceptual_cpp", &LearningModule::train_unsupervised_model_conceptual, "data_X"_a, "model_type_str"_a, "params"_a=py::dict())
        .def("train_ann_conceptual_cpp", &LearningModule::train_ann_conceptual, "data_X"_a, "data_y"_a, "ann_type_str"_a, "params"_a=py::dict())
        .def("train_autoencoder_conceptual_cpp", &LearningModule::train_autoencoder_conceptual, "data_X"_a, "encoding_dim"_a)
        .def("predict_with_model_stub_cpp", &LearningModule::predict_with_model_stub, "model_id"_a, "input_data"_a)
        .def("train_ess_vulnerability_predictor_stub_cpp", &LearningModule::train_ess_vulnerability_predictor_stub, "training_data_sim"_a, "model_config_params"_a=py::dict())
        .def("predict_vulnerability_for_ess_stub_cpp", &LearningModule::predict_vulnerability_for_ess_stub, "model_id"_a, "mutation_features"_a, "scenario_features"_a, "context_features_gs"_a)
        .def("featurize_mutation_for_ess_model_stub_cpp", &LearningModule::featurize_mutation_for_ess_model_stub, "mc_data_sim"_a)
        .def("featurize_scenario_config_for_ess_model_stub_cpp", &LearningModule::featurize_scenario_config_for_ess_model_stub, "scenario_cfg_data_sim"_a)
        .def("featurize_system_context_for_ess_model_stub_cpp", &LearningModule::featurize_system_context_for_ess_model_stub, "gs_snapshot_sim"_a)
        .def_property_readonly("last_lstm_loss_cpp", &LearningModule::get_last_lstm_loss)
        .def_property_readonly("last_q_reward_cpp", &LearningModule::get_last_q_reward)
        .def_property_readonly("ml_models_count_cpp", &LearningModule::get_ml_models_count)
        .def_property_readonly("learnings_in_kb_count_cpp", &LearningModule::get_learnings_in_kb_count);

    // SelfEvolutionModule (Bindings from Part 8)
    py::class_<IndividualCpp>(m, "IndividualCpp")
        .def(py::init<int>(), "genome_dim"_a)
        .def(py::init<const Eigen::VectorXd&>(), "parameters"_a)
        .def_readwrite("parameters", &IndividualCpp::parameters)
        .def_readwrite("fitness", &IndividualCpp::fitness)
        .def_readwrite("novelty_score", &IndividualCpp::novelty_score)
        .def_readwrite("age_generations", &IndividualCpp::age_generations)
        // .def_readwrite("parent_ids_sim", &IndividualCpp::parent_ids_sim) // std::optional needs careful binding or custom type caster
        .def_readwrite("id", &IndividualCpp::id);

    py::class_<ObjectiveDefinitionCpp>(m, "ObjectiveDefinitionCpp")
        .def(py::init<>())
        .def_readwrite("metric_path", &ObjectiveDefinitionCpp::metric_path)
        .def_readwrite("weight", &ObjectiveDefinitionCpp::weight)
        .def_readwrite("goal", &ObjectiveDefinitionCpp::goal)
        .def_readwrite("target_value", &ObjectiveDefinitionCpp::target_value)
        .def_readwrite("tolerance", &ObjectiveDefinitionCpp::tolerance)
        .def_readwrite("invert_for_fitness", &ObjectiveDefinitionCpp::invert_for_fitness)
        .def_readwrite("is_primary", &ObjectiveDefinitionCpp::is_primary);

    py::class_<FitnessLandscapeConfigCpp>(m, "FitnessLandscapeConfigCpp")
        .def(py::init<>())
        .def_readwrite("config_id", &FitnessLandscapeConfigCpp::config_id)
        .def_readwrite("description", &FitnessLandscapeConfigCpp::description)
        .def_readwrite("objective_definitions", &FitnessLandscapeConfigCpp::objective_definitions) // std::vector<ObjectiveDefinitionCpp>
        .def_readwrite("novelty_search_weight", &FitnessLandscapeConfigCpp::novelty_search_weight)
        .def_readwrite("creation_timestamp", &FitnessLandscapeConfigCpp::creation_timestamp)
        .def_readwrite("source_directive_sim", &FitnessLandscapeConfigCpp::source_directive_sim);
        
    py::class_<SelfEvolutionModule>(m, "SelfEvolutionModuleCpp")
        .def(py::init<CoreInterface*, int, double, double, int, int, int>(), "core_interface"_a,
             "population_size"_a=20, "mutation_rate_base"_a=0.1, "crossover_rate"_a=0.7,
             "novelty_archive_size"_a=100, "novelty_k_neighbors"_a=15, "abstract_genome_dim"_a=50)
        .def("update_logic_cpp", &SelfEvolutionModule::update_logic)
        .def("set_active_fitness_landscape_cpp", &SelfEvolutionModule::set_active_fitness_landscape, "new_landscape_config"_a)
        .def("get_active_fitness_landscape_cpp", &SelfEvolutionModule::get_active_fitness_landscape)
        .def_property_readonly("best_fitness_cpp", &SelfEvolutionModule::get_best_fitness)
        .def_property_readonly("average_fitness_cpp", &SelfEvolutionModule::get_average_fitness)
        .def_property_readonly("average_novelty_cpp", &SelfEvolutionModule::get_average_novelty)
        .def_property_readonly("current_generation_count_cpp", &SelfEvolutionModule::get_current_generation_count)
        .def_property_readonly("current_population_size_cpp", &SelfEvolutionModule::get_population_size_current);

    // FreeWillModule (Bindings from Part 8)
    py::class_<DecisionOptionCpp>(m, "DecisionOptionCpp") // Re-define if not already bound via FWE
        .def(py::init<int, int>(), "id"_a=0, "feature_dim"_a=0)
        .def_readwrite("id", &DecisionOptionCpp::id)
        .def_readwrite("features", &DecisionOptionCpp::features)
        .def_readwrite("value_score", &DecisionOptionCpp::value_score)
        .def_readwrite("goal_score", &DecisionOptionCpp::goal_score);

    py::class_<FreeWillModule>(m, "FreeWillModuleCpp")
        .def(py::init<CoreInterface*, int, int, double, double>(), "core_interface"_a,
             "num_options"_a=10, "feature_dim"_a=5, "beta"_a=5.0, "sigma"_a=0.1)
        .def("update_logic_cpp", &FreeWillModule::update_logic)
        .def("get_last_probabilities_cpp", &FreeWillModule::get_last_probabilities);

    // FreeWillEngine (Bindings from Part 9)
    py::class_<FreeWillEngine>(m, "FreeWillEngineCpp")
        .def(py::init<CoreInterface*, int, int, double, double, double, double, double>(), "core_interface"_a,
            "num_actions"_a = 10, "state_dim_env"_a = 5, 
            "alpha_lr"_a = 0.1, "gamma_discount"_a = 0.9, 
            "epsilon_start"_a = 0.8, "epsilon_end"_a = 0.05, "epsilon_decay_rate"_a = 0.001)
        .def("update_logic_cpp", &FreeWillEngine::update_logic) // This is the internal C++ loop
        .def("process_freewill_module_output_for_engine_cpp", &FreeWillEngine::process_freewill_module_output, // For Python FWM to call
            "options_from_fwm"_a, "probabilities_from_fwm"_a)
        .def_property_readonly("last_selected_action_id_cpp", &FreeWillEngine::get_last_selected_action_id)
        .def_property_readonly("last_total_reward_cpp", &FreeWillEngine::get_last_total_reward)
        .def_property_readonly("q_table_size_cpp", &FreeWillEngine::get_q_table_size)
        .def_property_readonly("current_epsilon_cpp", &FreeWillEngine::get_current_epsilon);

    // --- Interface & Support Modules ---
    // AdaptiveFirewallModule (Bindings from Part 9)
    // FirewallRuleCpp - need to handle std::optional and std::regex carefully
    py::class_<FirewallRuleCpp>(m, "FirewallRuleCpp")
        .def(py::init<>())
        .def_readwrite("id", &FirewallRuleCpp::id)
        .def_readwrite("action", &FirewallRuleCpp::action)
        .def_property("src_ip_pattern_str", [](const FirewallRuleCpp& r){ return r.src_ip_pattern_str; }, [](FirewallRuleCpp& r, const std::optional<std::string>& s){ r.src_ip_pattern_str = s; })
        .def_property("dst_ip_pattern_str", [](const FirewallRuleCpp& r){ return r.dst_ip_pattern_str; }, [](FirewallRuleCpp& r, const std::optional<std::string>& s){ r.dst_ip_pattern_str = s; })
        .def_property("src_port", [](const FirewallRuleCpp& r){ return r.src_port; }, [](FirewallRuleCpp& r, const std::optional<int>& p){ r.src_port = p; })
        .def_property("dst_port", [](const FirewallRuleCpp& r){ return r.dst_port; }, [](FirewallRuleCpp& r, const std::optional<int>& p){ r.dst_port = p; })
        .def_property("protocol", [](const FirewallRuleCpp& r){ return r.protocol; }, [](FirewallRuleCpp& r, const std::optional<std::string>& p){ r.protocol = p; })
        .def_readwrite("payload_regex_str_original", &FirewallRuleCpp::payload_regex_str_original)
        .def("compile_payload_regex_cpp", &FirewallRuleCpp::compile_payload_regex)
        .def_property("min_threat_score", [](const FirewallRuleCpp& r){ return r.min_threat_score; }, [](FirewallRuleCpp& r, const std::optional<double>& s){ r.min_threat_score = s; })
        .def_readwrite("priority", &FirewallRuleCpp::priority)
        .def_readwrite("description", &FirewallRuleCpp::description)
        .def_readwrite("enabled", &FirewallRuleCpp::enabled)
        .def_property_readonly("hit_count_cpp", [](const FirewallRuleCpp& r){ return r.hit_count; })
        .def_readwrite("last_hit_timestamp", &FirewallRuleCpp::last_hit_timestamp)
        .def_readwrite("created_at_timestamp", &FirewallRuleCpp::created_at_timestamp)
        .def_readwrite("learned_by", &FirewallRuleCpp::learned_by)
        .def_readwrite("tags_vec", &FirewallRuleCpp::tags_vec);
        // .def("matches_py_dict", [](const FirewallRuleCpp& self, py::dict packet_info_py) { /* Convert dict to map and call matches */ }) // Example

    py::class_<TrafficFeatureVectorCpp>(m, "TrafficFeatureVectorCpp")
        .def(py::init<>())
        .def_readwrite("timestamp", &TrafficFeatureVectorCpp::timestamp)
        .def_readwrite("src_ip", &TrafficFeatureVectorCpp::src_ip)
        .def_readwrite("dst_ip", &TrafficFeatureVectorCpp::dst_ip)
        .def_readwrite("src_port", &TrafficFeatureVectorCpp::src_port)
        .def_readwrite("dst_port", &TrafficFeatureVectorCpp::dst_port)
        .def_readwrite("protocol", &TrafficFeatureVectorCpp::protocol)
        .def_readwrite("packet_size", &TrafficFeatureVectorCpp::packet_size)
        .def_readwrite("payload_entropy", &TrafficFeatureVectorCpp::payload_entropy);
        
    py::class_<AdaptiveFirewallModule>(m, "AdaptiveFirewallModuleCpp")
        .def(py::init<CoreInterface*>(), "core_interface"_a)
        .def("update_logic_cpp", &AdaptiveFirewallModule::update_logic)
        .def("add_rule_cpp", &AdaptiveFirewallModule::add_rule, "rule_data"_a)
        .def("remove_rule_cpp", &AdaptiveFirewallModule::remove_rule, "rule_id"_a)
        .def("update_rule_cpp", &AdaptiveFirewallModule::update_rule, "rule_id"_a, "new_rule_data"_a)
        .def("get_rule_cpp", &AdaptiveFirewallModule::get_rule, "rule_id"_a)
        .def("get_all_rules_cpp", &AdaptiveFirewallModule::get_all_rules)
        .def("train_threat_model_stub_cpp", &AdaptiveFirewallModule::train_threat_model_stub, "benign_sample"_a, "malicious_sample"_a)
        .def("predict_threat_score_stub_cpp", &AdaptiveFirewallModule::predict_threat_score_stub, "traffic_features"_a);

    // TimeSeriesPredictorModule (Bindings from Part 9)
    py::class_<TimeSeriesDataCpp>(m, "TimeSeriesDataCpp")
        .def(py::init<std::string>(), "series_id"_a = "")
        .def_readwrite("id", &TimeSeriesDataCpp::id)
        .def_readwrite("timestamps", &TimeSeriesDataCpp::timestamps)
        .def_readwrite("values", &TimeSeriesDataCpp::values)
        .def_readwrite("metadata", &TimeSeriesDataCpp::metadata)
        .def_readwrite("model_type_id", &TimeSeriesDataCpp::model_type_id)
        // model_parameters_internal_stub (std::any) is hard to bind directly.
        // last_prediction_result_stub (std::optional<map<string, Eigen::VectorXd>>) also needs care.
        .def_readwrite("prediction_default_horizon", &TimeSeriesDataCpp::prediction_default_horizon)
        .def_readwrite("max_length", &TimeSeriesDataCpp::max_length);

    py::class_<TimeSeriesPredictorModule>(m, "TimeSeriesPredictorModuleCpp")
        .def(py::init<CoreInterface*>(), "core_interface"_a)
        .def("update_logic_cpp", &TimeSeriesPredictorModule::update_logic)
        .def("add_or_update_data_point_cpp", &TimeSeriesPredictorModule::add_or_update_data_point, "series_id"_a, "timestamp"_a, "value"_a)
        .def("register_or_update_series_definition_cpp", &TimeSeriesPredictorModule::register_or_update_series_definition, "ts_definition"_a)
        .def("train_model_for_series_stub_cpp", &TimeSeriesPredictorModule::train_model_for_series_stub, "series_id"_a, "model_type_to_use"_a, "training_params"_a = py::dict())
        .def("predict_series_stub_cpp", &TimeSeriesPredictorModule::predict_series_stub, "series_id"_a, "horizon_override"_a = std::nullopt)
        .def("get_time_series_data_copy_cpp", &TimeSeriesPredictorModule::get_time_series_data_copy, "series_id"_a)
        .def("get_all_series_ids_cpp", &TimeSeriesPredictorModule::get_all_series_ids);

    // --- Advanced Processing & Lyuk ---
    // ControlledMutationGenerator (Bindings from Part 7 & 10)
    // MuGenParameterValueCpp (std::variant) requires custom type casters or wrapper functions for Python.
    // For simplicity, methods taking/returning complex variants might be exposed via simpler types or dicts.
    py::class_<MutationCandidateCpp>(m, "MutationCandidateCpp")
        .def(py::init<>())
        .def_readwrite("candidate_id", &MutationCandidateCpp::candidate_id)
        .def_readwrite("target_type_str", &MutationCandidateCpp::target_type_str)
        .def_readwrite("target_identifier_str", &MutationCandidateCpp::target_identifier_str)
        .def_property("parameter_name_opt_str", [](const MutationCandidateCpp& o){ return o.parameter_name_opt_str; }, [](MutationCandidateCpp& o, const std::optional<std::string>& s){ o.parameter_name_opt_str = s; } )
        // .def_readwrite("original_value_variant", &MutationCandidateCpp::original_value_variant) // Hard to bind variant
        // .def_readwrite("mutated_value_variant", &MutationCandidateCpp::mutated_value_variant)   // Hard to bind variant
        .def("get_original_value_str_sim", [](const MutationCandidateCpp& c){ return mugen_param_value_to_string(c.original_value_variant); }) // Expose as string
        .def("get_mutated_value_str_sim", [](const MutationCandidateCpp& c){ return mugen_param_value_to_string(c.mutated_value_variant); })   // Expose as string
        .def_readwrite("mutation_operator_used_str", &MutationCandidateCpp::mutation_operator_used_str)
        .def_readwrite("predicted_impact_vector_sim_map", &MutationCandidateCpp::predicted_impact_vector_sim_map)
        .def_readwrite("overall_predicted_desirability_sim_val", &MutationCandidateCpp::overall_predicted_desirability_sim_val)
        .def_readwrite("simulation_confidence_sim_val", &MutationCandidateCpp::simulation_confidence_sim_val)
        .def_readwrite("meets_improvement_threshold_sim_flag", &MutationCandidateCpp::meets_improvement_threshold_sim_flag)
        .def_readwrite("timestamp_val", &MutationCandidateCpp::timestamp_val)
        .def_property("context_hash_at_proposal_sim_str", [](const MutationCandidateCpp& o){ return o.context_hash_at_proposal_sim_str; }, [](MutationCandidateCpp& o, const std::optional<std::string>& s){ o.context_hash_at_proposal_sim_str = s; } )
        .def_readwrite("tags_vec", &MutationCandidateCpp::tags_vec);

    py::class_<ControlledMutationGenerator>(m, "ControlledMutationGeneratorCpp")
        .def(py::init<CoreInterface*, int>(), "core_interface"_a, "abstract_genome_dim_hint"_a=50)
        .def("update_logic_cpp", &ControlledMutationGenerator::update_logic)
        .def("generate_and_propose_mutation_from_abstract_genome_py_dict_context", // Renamed to clarify Python dict context
            [](ControlledMutationGenerator& self, const Eigen::VectorXd& abstract_genome, py::dict sem_context_py) {
                std::map<std::string, std::any> sem_context_cpp; // Convert py::dict to std::map<std::string, std::any>
                for(auto item : sem_context_py) { // Basic conversion, needs error handling and more types
                    std::string key = item.first.cast<std::string>();
                    if (py::isinstance<py::str>(item.second)) sem_context_cpp[key] = item.second.cast<std::string>();
                    else if (py::isinstance<py::float_>(item.second) || py::isinstance<py::int_>(item.second)) sem_context_cpp[key] = item.second.cast<double>();
                    // else if (py::isinstance<py::array_t<double>>(item.second)) // For Eigen or vector
                }
                return self.generate_and_propose_mutation_from_abstract_genome(abstract_genome, sem_context_cpp);
            }, "abstract_genome"_a, "sem_context_py_dict"_a, "Generates mutation from genome using Python dict context.")
        .def("evaluate_specific_mutation_candidate_with_surrogate_cpp", &ControlledMutationGenerator::evaluate_specific_mutation_candidate_with_surrogate,
             "concrete_mutation_template"_a, "current_gs_for_context"_a, "Evaluates a concrete mutation.");


    // LyukParser (Bindings from Part 7 & 10)
    // Binding AST nodes is complex. Expose functions that return serializable representations (e.g., stringified AST).
    py::class_<LyukParser>(m, "LyukParserCpp")
        .def(py::init<CoreInterface*>(), "core_interface"_a)
        .def("parse_lyuk_code_to_ast_summary_str_stub", // Returns a string summary
            [](LyukParser& self, const std::string& lyuk_code) {
                auto ast_nodes = self.parse_lyuk_code_to_ast_stub(lyuk_code);
                std::string summary = "Parsed Lyuk AST (C++ Stub, " + std::to_string(ast_nodes.size()) + " top nodes):\n";
                for(size_t i=0; i < std::min(ast_nodes.size(), static_cast<size_t>(5)); ++i) { // Show first 5
                    if(ast_nodes[i]) summary += "  - Node " + std::to_string(i) + ": " + ast_nodes[i]->primitive_name_str + " (" + ast_nodes[i]->raw_arguments_original_text_str.substr(0,30) + "...)\n";
                }
                return summary;
            }, "lyuk_code_str"_a)
        .def("interpret_ast_semantically_summary_str_stub", // Returns string summary of interpretation
            [](LyukParser& self, const std::string& lyuk_code) { // Takes code, parses, then interprets
                auto ast_nodes = self.parse_lyuk_code_to_ast_stub(lyuk_code);
                auto interpretation_map_any = self.interpret_ast_semantically_stub(ast_nodes);
                std::string summary = "Semantic Interpretation (C++ Stub):\n";
                for(const auto& pair_item : interpretation_map_any){ // std::any needs careful handling
                    summary += "  - " + pair_item.first + ": <value_type:" + pair_item.second.type().name() + ">\n";
                }
                return summary;
            }, "lyuk_code_str"_a)
        .def("transcompile_ast_to_human_readable_stub_from_code",
            [](LyukParser& self, const std::string& lyuk_code, const std::string& target_repr_type) {
                auto ast_nodes = self.parse_lyuk_code_to_ast_stub(lyuk_code);
                return self.transcompile_ast_to_human_readable_stub(ast_nodes, target_repr_type);
            }, "lyuk_code_str"_a, "target_representation_type_str"_a = "pseudocode_cpp_detail_stub");


    // --- Homeostasis & Regulation Modules (Stubs - Bindings from Part 7, 8, 9) ---
    py::class_<ConsciousStateCpp>(m, "ConsciousStateCpp") // If not bound before
        .def(py::init<int,int,int>())
        .def_readwrite("perception_vec", &ConsciousStateCpp::perception_vec)
        .def_readwrite("decision_vec", &ConsciousStateCpp::decision_vec)
        .def_readwrite("narrative_vec", &ConsciousStateCpp::narrative_vec);

    py::class_<ConsciousnessModuleCpp>(m, "ConsciousnessModuleCpp_Stub")
        .def(py::init<CoreInterface*, int, int, int, double>(), "core_interface"_a, 
             "perception_dim"_a=10, "decision_dim"_a=3, "narrative_dim"_a=5, "phi_modulation_factor"_a=0.15)
        .def("update_logic_cpp", &ConsciousnessModuleCpp::update_logic)
        .def("process_shimyureshon_reflexion_results_stub_py_dict", // For Python to call with dict
            [](ConsciousnessModuleCpp& self, const std::string& sh_id, py::dict report_py_dict){
                std::map<std::string, std::any> report_cpp_map; // Convert dict to map<string, any>
                for(auto item : report_py_dict) { /* Basic conversion */ report_cpp_map[item.first.cast<std::string>()] = item.second; }
                self.process_shimyureshon_reflexion_results_stub(sh_id, report_cpp_map);
            }, "sh_id_str"_a, "report_py_dict"_a);

    py::class_<EmotionStateDataCpp>(m, "EmotionStateDataCpp") // If not bound before
        .def(py::init<>())
        .def_readwrite("valence", &EmotionStateDataCpp::valence)
        .def_readwrite("arousal", &EmotionStateDataCpp::arousal);

    py::class_<EmotionRegulationModuleCpp>(m, "EmotionRegulationModuleCpp_Stub")
        .def(py::init<CoreInterface*, double, double, double, double, double, double>(), "core_interface"_a,
             "reference_valence"_a=0.15, "reference_arousal"_a=0.4, 
             "kp"_a=0.3, "ki"_a=0.06, "kd"_a=0.03, "dt_factor_erm"_a=1.0)
        .def("update_logic_cpp", &EmotionRegulationModuleCpp::update_logic);

    py::class_<NeedsManagerCpp>(m, "NeedsManagerCpp_Stub")
        .def(py::init<CoreInterface*, double>(), "core_interface"_a, "dt_factor"_a = 1.0)
        .def("update_logic_cpp", &NeedsManagerCpp::update_logic)
        .def_property_readonly("current_need_priorities_cpp", &NeedsManagerCpp::get_current_need_priorities)
        .def_property_readonly("last_proposed_action_name_cpp", &NeedsManagerCpp::get_last_proposed_action_name);

    py::class_<CravingModuleCpp>(m, "CravingModuleCpp_Stub")
        .def(py::init<CoreInterface*, int, double, double>(), "core_interface"_a,
             "num_cravings"_a = 3, "alpha"_a = 0.6, "beta"_a = 0.4)
        .def("update_logic_cpp", &CravingModuleCpp::update_logic)
        .def_property_readonly("current_intensities_cpp", &CravingModuleCpp::get_current_intensities)
        .def_property_readonly("craving_names_cpp", &CravingModuleCpp::get_craving_names);

    py::class_<SelfCompassionModuleCpp>(m, "SelfCompassionModuleCpp_Stub")
        .def(py::init<CoreInterface*>(), "core_interface"_a)
        .def("update_logic_cpp", &SelfCompassionModuleCpp::update_logic)
        .def_property_readonly("internal_compassion_score_cpp", &SelfCompassionModuleCpp::get_internal_compassion_score)
        .def_property_readonly("is_recovery_mode_active_cpp", &SelfCompassionModuleCpp::is_recovery_mode_active);

    py::class_<StressResponseModuleCpp>(m, "StressResponseModuleCpp_Stub")
        .def(py::init<CoreInterface*>(), "core_interface"_a)
        .def("update_logic_cpp", &StressResponseModuleCpp::update_logic)
        .def_property_readonly("current_stress_level_cpp", &StressResponseModuleCpp::get_current_stress_level);

    py::class_<PainMatrixDirectiveCpp>(m, "PainMatrixDirectiveCpp_Stub")
        .def(py::init<CoreInterface*>(), "core_interface"_a)
        .def("update_logic_cpp", &PainMatrixDirectiveCpp::update_logic)
        .def_property_readonly("current_pain_level_cpp", &PainMatrixDirectiveCpp::get_current_pain_level);

    py::class_<DefenseMechanismsCpp>(m, "DefenseMechanismsCpp_Stub")
        .def(py::init<CoreInterface*>(), "core_interface"_a)
        .def("update_logic_cpp", &DefenseMechanismsCpp::update_logic)
        .def_property_readonly("active_defense_stub_cpp", &DefenseMechanismsCpp::get_active_defense_stub);

    // --- Communication & Social Stubs ---
    py::class_<IlyukMessageStructureCpp>(m, "IlyukMessageStructureCpp") // Re-define if not bound before
        .def(py::init<>())
        .def_readwrite("campo_emocional", &IlyukMessageStructureCpp::campo_emocional)
        .def_readwrite("campo_logico", &IlyukMessageStructureCpp::campo_logico)
        .def_readwrite("campo_ontologico_intencional", &IlyukMessageStructureCpp::campo_ontologico_intencional)
        .def_readwrite("metadata", &IlyukMessageStructureCpp::metadata) // map<string,string>
        .def_readwrite("message_id", &IlyukMessageStructureCpp::message_id);

    py::class_<LlyukCommunicationModuleCpp>(m, "LlyukCommunicationModuleCpp_Stub")
        .def(py::init<CoreInterface*>(), "core_interface"_a)
        .def("update_logic_cpp", &LlyukCommunicationModuleCpp::update_logic)
        .def("send_lyuk_message_stub_cpp", &LlyukCommunicationModuleCpp::send_lyuk_message_stub, "message"_a, "target_entity_id_sim"_a);
        // process_incoming_interpreted_lyuk_stub_py was defined in Part 7/8, ensure it's here.

    py::class_<SocialAgentCpp>(m, "SocialAgentCpp") // Re-define if not bound before
        .def(py::init<>())
        .def_readwrite("id", &SocialAgentCpp::id)
        .def_readwrite("type", &SocialAgentCpp::type)
        // attributes and inferred_mental_state are std::map<std::string, std::any>
        .def_readwrite("last_interaction_timestamp", &SocialAgentCpp::last_interaction_timestamp);
        
    py::class_<SocialInteractionCpp>(m, "SocialInteractionCpp") // Re-define if not bound
        .def(py::init<>())
        .def_readwrite("id", &SocialInteractionCpp::id)
        .def_readwrite("timestamp", &SocialInteractionCpp::timestamp)
        .def_readwrite("initiator_id", &SocialInteractionCpp::initiator_id)
        .def_readwrite("target_id", &SocialInteractionCpp::target_id)
        .def_readwrite("type", &SocialInteractionCpp::type)
        .def_readwrite("content_summary", &SocialInteractionCpp::content_summary)
        .def_readwrite("outcome", &SocialInteractionCpp::outcome);

    py::class_<SocialDynamicsModuleCpp>(m, "SocialDynamicsModuleCpp_Stub")
        .def(py::init<CoreInterface*>(), "core_interface"_a)
        .def("update_logic_cpp", &SocialDynamicsModuleCpp::update_logic)
        .def("add_or_update_agent_stub_cpp", &SocialDynamicsModuleCpp::add_or_update_agent_stub, "agent_data"_a)
        .def("record_interaction_stub_cpp", &SocialDynamicsModuleCpp::record_interaction_stub, "interaction_data"_a)
        .def("get_agent_model_stub_cpp", &SocialDynamicsModuleCpp::get_agent_model_stub, "agent_id"_a);
    
    py::class_<OntologyFlowManagerCpp>(m, "OntologyFlowManagerCpp_Stub")
        .def(py::init<CoreInterface*>(), "core_interface"_a)
        .def("update_logic_cpp", &OntologyFlowManagerCpp::update_logic)
        .def("get_current_hypotheses_count_stub_cpp", 
             [](const OntologyFlowManagerCpp& self) { return self.get_current_hypotheses_stub().size(); });


    // --- Module Version ---
    m.attr("__version__") = "eane_cpp_core_v16.0_phoenix_full_suite_bindings";
} // End PYBIND11_MODULE
