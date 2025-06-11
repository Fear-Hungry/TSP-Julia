# --- Ativação do Ambiente e Dependências ---
# Garante que o ambiente do projeto seja ativado e que todas as
# dependências listadas em Project.toml estejam instaladas.
import Pkg
Pkg.activate(".")
Pkg.instantiate()
# ------------------------------------------

using ArgParse
using Plots
using Printf
using Logging
using Distributed

# Configura o Logger para exibir mensagens de @info e acima.
global_logger(SimpleLogger(stdout, Logging.Info))

# --- Inclusão dos Módulos do Projeto ---
# Usamos `include` para carregar os arquivos dos módulos e `using` para trazer
# suas funções para o escopo atual.
include("genetic/Utils.jl")
using .UtilsModule

include("genetic/Individual.jl")
using .IndividualModule

include("genetic/Optimizer.jl")
using .OptimizerModule

include("genetic/Visualization.jl")
using .VisualizationModule
# ----------------------------------------

const DEFAULT_PARAMETERS = Dict(
    "modo" => "ucb1",
    "ucb_c" => 2.0,
    "arquivo" => "Qatar.txt",
    "geracoes" => 1000,
    "tam_populacao" => 100,
    "tam_elite" => 0.1,
    "taxa_mutacao" => 0.01,
    "taxa_crossover" => 0.8,
    "taxa_2opt" => 0.1,
    "k_neighbors" => 20
)

function create_argument_parser()
    parser = ArgParseSettings(description="Algoritmo Genético para TSP em Julia")

    @add_arg_table! parser begin
        "--modo"
            help = "Modo: 'padrao' (taxas fixas) ou 'ucb1' (seleção adaptativa)"
            arg_type = String
            default = DEFAULT_PARAMETERS["modo"]
        "--ucb_c"
            help = "Constante de exploração UCB1"
            arg_type = Float64
            default = DEFAULT_PARAMETERS["ucb_c"]
        "--arquivo"
            help = "Arquivo de entrada (pasta 'EntradasTSP/')"
            arg_type = String
            default = DEFAULT_PARAMETERS["arquivo"]
        "--geracoes"
            help = "Número de gerações"
            arg_type = Int
            default = DEFAULT_PARAMETERS["geracoes"]
        "--tam_populacao"
            help = "Tamanho da população"
            arg_type = Int
            default = DEFAULT_PARAMETERS["tam_populacao"]
        "--tam_elite"
            help = "Proporção da elite (0.1 = 10%)"
            arg_type = Float64
            default = DEFAULT_PARAMETERS["tam_elite"]
        "--taxa_mutacao"
            help = "Taxa de mutação por troca"
            arg_type = Float64
            default = DEFAULT_PARAMETERS["taxa_mutacao"]
        "--taxa_crossover"
            help = "Taxa de crossover"
            arg_type = Float64
            default = DEFAULT_PARAMETERS["taxa_crossover"]
        "--taxa_2opt"
            help = "Taxa de melhoria 2-opt"
            arg_type = Float64
            default = DEFAULT_PARAMETERS["taxa_2opt"]
        "--k_neighbors"
            help = "Vizinhos para matriz esparsa (0 = matriz densa)"
            arg_type = Int
            default = DEFAULT_PARAMETERS["k_neighbors"]
        "--interactive"
            help = "Visualização interativa"
            action = :store_true
        "--animation"
            help = "Gerar GIF animado"
            action = :store_true
    end

    return parser
end

function validate_input_file(file_path::String)
    if !isfile(file_path)
        @error "Arquivo não encontrado: $file_path"
        return false
    end
    return true
end

function create_optimizer_from_parameters(parameters::Dict, problem_data)
    n, coordinates, distance_matrix = problem_data

    return OptimizerModule.GeneticOptimizer(
        population_size = parameters["tam_populacao"],
        elite_size_ratio = parameters["tam_elite"],
        generations = parameters["geracoes"],
        n_cities = n,
        coords = coordinates,
        dist_matrix = distance_matrix,
        aos_mode = parameters["modo"] == "ucb1" ? "ucb1" : nothing,
        ucb_c = parameters["ucb_c"],
        mutation_rate = parameters["taxa_mutacao"],
        crossover_rate = parameters["taxa_crossover"],
        two_opt_rate = parameters["taxa_2opt"]
    )
end

function prepare_problem_data(parameters::Dict)
    @info "Carregando dados do TSP: $(basename(parameters["arquivo"])))"

        n, x_coords, y_coords = read_tsp_file(parameters["arquivo"])
    coordinates = (x_coords, y_coords)

    distance_matrix = create_distance_matrix(n, x_coords, y_coords, k=parameters["k_neighbors"])

    if parameters["k_neighbors"] > 0
        @info "Matriz esparsa criada com k=$(parameters["k_neighbors"]) vizinhos"
    end

    return n, coordinates, distance_matrix
end

function execute_genetic_algorithm(parameters::Dict)
    start_time = time()

    problem_data = prepare_problem_data(parameters)
    optimizer = create_optimizer_from_parameters(parameters, problem_data)

    initialize_population!(optimizer)
    best_route, best_distance, convergence_history = evolve!(optimizer)

    execution_time = time() - start_time

    log_results(best_route, best_distance, execution_time, problem_data[1])

    if !parameters["interactive"]
        save_results(parameters, best_route, best_distance, convergence_history, problem_data)
    end

    return best_route, best_distance, convergence_history
end

function log_results(route::Vector{Int}, distance::Float32, time_elapsed::Float64, num_cities::Int)
    if is_valid_tsp_solution(route, num_cities)
        @info "Solução VÁLIDA encontrada"
    else
        @warn "Solução INVÁLIDA encontrada"
    end

    @info @sprintf("Melhor distância: %.2f", distance)
    @info @sprintf("Tempo de execução: %.2f segundos", time_elapsed)
end

function save_results(parameters::Dict, route::Vector{Int}, distance::Float32,
                     convergence::Dict, problem_data)
    _, coordinates, _ = problem_data
    x_coords, y_coords = coordinates

    base_filename = split(basename(parameters["arquivo"]), '.')[1]
    result_path = "resultados/ga/$(base_filename)"

    plot_route(x_coords, y_coords, route,
               "Melhor Rota - $(basename(parameters["arquivo"]))",
               distance,
               save_path="$(result_path)_rota.png")

    plot_convergence(convergence["generations"], convergence["distances"],
                     title="Convergência - $(basename(parameters["arquivo"]))",
                     save_path="$(result_path)_convergencia.png")
end

function main()
    parser = create_argument_parser()
    parameters = parse_args(parser)

    mkpath("resultados/ga")

    full_file_path = joinpath("EntradasTSP", parameters["arquivo"])
    if !validate_input_file(full_file_path)
        return
    end

    parameters["arquivo"] = full_file_path
    execute_genetic_algorithm(parameters)
end

# Ponto de entrada do script, executa main() se o arquivo for chamado diretamente.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
