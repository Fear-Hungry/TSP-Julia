module VisualizationModule

export plot_route, plot_convergence, create_ga_animation, InteractiveGAVisualization, update!, finalize!

using Plots
using Printf
using StatsBase

# Configura o backend de plots para ser executado em ambientes sem GUI
# pyplot()

# --- Funções de Plotagem Estática ---

"""
    plot_route(x, y, route, title, distance; save_path=nothing)

Plota a rota do TSP, incluindo anotações para as cidades.
"""
function plot_route(x::Vector{Float32}, y::Vector{Float32}, route::Vector{Int},
                    title::String, distance::Float32; save_path::Union{String, Nothing}=nothing)

    # Adiciona o ponto inicial ao final para fechar o ciclo
    route_to_plot = [route; route[1]]

    plot_x = x[route_to_plot]
    plot_y = y[route_to_plot]

    p = plot(plot_x, plot_y,
             marker=(:circle, 5),
             linecolor=:dimgray,
             label="Rota",
             xlabel="Coordenada X",
             ylabel="Coordenada Y",
             framestyle=:box,
             legend=false)

    # Adiciona as cidades e suas anotações
    scatter!(p, x, y, markercolor=:skyblue, markersize=6, label="Cidades")
    for i in 1:length(x)
        annotate!(p, x[i], y[i], text(string(i), :gray, :left, 8))
    end

    # Destaca o ponto inicial
    scatter!(p, [x[route[1]]], [y[route[1]]], marker=:star, color=:gold, markersize=10)

    full_title = "$title\nDistância: $(@sprintf("%.2f", distance))"
    title!(p, full_title)

    if save_path !== nothing
        mkpath(dirname(save_path))
        savefig(p, save_path)
        @info "Gráfico da rota salvo em: $save_path"
    else
        display(p)
    end
    return p
end

"""
    plot_convergence(generations, distances; title="Convergência", save_path=nothing)

Plota o gráfico de convergência, destacando o melhor ponto.
"""
function plot_convergence(generations::Vector{Int}, distances::Vector{Float32};
                          title::String="Convergência", save_path::Union{String, Nothing}=nothing)

    p = plot(generations, distances,
        title=title,
        xlabel="Geração",
        ylabel="Melhor Distância",
        label="Distância",
        legend=:topright,
        grid=true)

    # Destaca o melhor ponto
    if !isempty(distances)
        min_dist, best_idx = findmin(distances)
        best_gen = generations[best_idx]
        scatter!(p, [best_gen], [min_dist], color=:red, label="Melhor Ponto", markersize=8)
        annotate!(p, best_gen, min_dist, text("Melhor: $(@sprintf("%.2f", min_dist))", :red, :left, 10))
    end

    if save_path !== nothing
        mkpath(dirname(save_path))
        savefig(p, save_path)
        @info "Gráfico de convergência salvo em $save_path"
    else
        display(p)
    end
    return p
end


# --- Visualização Interativa ---

"""
    InteractiveGAVisualization

Estrutura para gerenciar a visualização interativa do GA com dois subplots.
"""
mutable struct InteractiveGAVisualization
    x::Vector{Float32}
    y::Vector{Float32}
    update_interval::Int

    generations::Vector{Int}
    distances::Vector{Float32}
    best_route::Vector{Int}
    best_distance::Float32
    current_generation::Int

    fig::Plots.Plot # O objeto do plot principal que contém os subplots

    function InteractiveGAVisualization(x, y; title="Evolução do GA", update_interval=10)
        generations = Int[]
        distances = Float32[]
        best_route = Int[]
        best_distance = Inf32

        # Cria um layout vazio inicial
        p1 = plot(title="Melhor Rota", xlabel="X", ylabel="Y", grid=true)
        p2 = plot(title="Convergência", xlabel="Geração", ylabel="Distância", grid=true)
        fig = plot(p1, p2, layout=(1, 2), size=(1200, 500), title=title)

        new(x, y, update_interval, generations, distances, best_route, best_distance, 0, fig)
    end
end

"""
    update!(viz, route, distance, generation)

Atualiza os dados e, se for o caso, o plot da visualização interativa.
"""
function update!(viz::InteractiveGAVisualization, route::Vector{Int}, distance::Float32, generation::Int)
    viz.current_generation = generation
    push!(viz.generations, generation)
    push!(viz.distances, distance)

    if distance < viz.best_distance
        viz.best_distance = distance
        viz.best_route = copy(route)
    end

    if generation % viz.update_interval == 0 || generation == 1
        _update_plots!(viz)
    end
end

"""
    _update_plots!(viz)

Função interna para redesenhar os subplots da visualização.
"""
function _update_plots!(viz::InteractiveGAVisualization)
    # Subplot 1: Rota
    p1 = plot_route(viz.x, viz.y, viz.best_route, "Melhor Rota Atual", viz.best_distance)

    # Subplot 2: Convergência
    p2 = plot_convergence(viz.generations, viz.distances, title="Convergência do Algoritmo")

    # Combina os plots e atualiza a figura
    viz.fig = plot(p1, p2, layout=(1, 2), size=(1200, 500),
                   title="Geração: $(viz.current_generation) | Melhor Distância: $(@sprintf("%.2f", viz.best_distance))")
    display(viz.fig)
end


"""
    finalize!(viz, save_path) -> Dict

Finaliza a visualização, faz um último update e salva se solicitado.
"""
function finalize!(viz::InteractiveGAVisualization; save_path::Union{String, Nothing}=nothing)
    _update_plots!(viz) # Garante que o último estado seja exibido

    if save_path !== nothing
        mkpath(dirname(save_path))
        savefig(viz.fig, save_path)
    end

    return Dict(
        "best_route" => viz.best_route,
        "best_distance" => viz.best_distance,
        "generations" => viz.generations,
        "distances" => viz.distances
    )
end

# --- Animação ---
"""
    create_ga_animation(x, y, all_routes, all_distances, output_path)

Cria uma animação GIF da evolução do GA.
"""
function create_ga_animation(x, y, all_routes, all_distances, output_path)
    @info "Criando animação... Isso pode levar um momento."

    anim = @animate for i in 1:length(all_routes)
        route = all_routes[i]
        dist = all_distances[i]

        # Reutiliza a função de plot de rota para cada frame
        plot_route(x, y, route, "Geração $i", dist)
    end

    gif(anim, output_path, fps = 10)
    @info "Animação salva em: $output_path"
end


end # fim do módulo
