module UtilsModule

export read_tsp_file, create_distance_matrix, is_valid_tsp_solution

using DelimitedFiles
using NearestNeighbors
using Distances
using SparseArrays

struct TSPData
    num_cities::Int
    x_coordinates::Vector{Float32}
    y_coordinates::Vector{Float32}
end

function read_tsp_file(file_path::String)
    file_lines = readlines(file_path)
    num_cities = parse(Int, file_lines[1])

    coordinates_data = read_coordinates_from_lines(file_lines[2:end])
    x_coords, y_coords = extract_coordinate_vectors(coordinates_data, num_cities)

    return num_cities, x_coords, y_coords
end

function read_coordinates_from_lines(coordinate_lines::Vector{String})
    coordinates_buffer = IOBuffer(join(coordinate_lines, "\n"))
    return readdlm(coordinates_buffer)
end

function extract_coordinate_vectors(coordinates_data, num_cities::Int)
    x_coords = zeros(Float32, num_cities)
    y_coords = zeros(Float32, num_cities)

    for i in 1:size(coordinates_data, 1)
        city_index = Int(coordinates_data[i, 1])
        x_coords[city_index] = coordinates_data[i, 2]
        y_coords[city_index] = coordinates_data[i, 3]
    end

    return x_coords, y_coords
end

function create_distance_matrix(num_cities::Int, x_coords::Vector{Float32}, y_coords::Vector{Float32}; k::Int=0)
    coordinates_matrix = hcat(x_coords, y_coords)

    return should_use_sparse_matrix(k) ?
           create_sparse_distance_matrix(coordinates_matrix, k) :
           create_dense_distance_matrix(coordinates_matrix)
end

function should_use_sparse_matrix(k::Int)
    return k > 0
end

function create_dense_distance_matrix(coordinates_matrix)
    return pairwise(Euclidean(), coordinates_matrix', dims=2)
end

function create_sparse_distance_matrix(coordinates_matrix, k::Int)
    kdtree = KDTree(coordinates_matrix')
    neighbor_indices, neighbor_distances = find_k_nearest_neighbors(kdtree, coordinates_matrix', k)
    return build_sparse_matrix_from_neighbors(neighbor_indices, neighbor_distances, size(coordinates_matrix, 1))
end

function find_k_nearest_neighbors(kdtree, coordinates, k::Int)
    return knn(kdtree, coordinates, k + 1, true)
end

function build_sparse_matrix_from_neighbors(indices_list, distances_list, num_cities::Int)
    sparse_rows, sparse_cols, sparse_values = prepare_sparse_arrays(num_cities, length(indices_list[1]) - 1)

    populate_sparse_arrays!(sparse_rows, sparse_cols, sparse_values, indices_list, distances_list)

    return sparse(sparse_rows, sparse_cols, sparse_values, num_cities, num_cities)
end

function prepare_sparse_arrays(num_cities::Int, neighbors_per_city::Int)
    expected_size = num_cities * neighbors_per_city

    rows = Int[]
    cols = Int[]
    values = Float32[]

    sizehint!(rows, expected_size)
    sizehint!(cols, expected_size)
    sizehint!(values, expected_size)

    return rows, cols, values
end

function populate_sparse_arrays!(rows, cols, values, indices_list, distances_list)
    for (city_index, (neighbor_indices, neighbor_distances)) in enumerate(zip(indices_list, distances_list))
        add_city_neighbors!(rows, cols, values, city_index, neighbor_indices, neighbor_distances)
    end
end

function add_city_neighbors!(rows, cols, values, city_index::Int, neighbor_indices, neighbor_distances)
    for (neighbor_index, distance) in zip(neighbor_indices, neighbor_distances)
        if city_index != neighbor_index
            push!(rows, city_index)
            push!(cols, neighbor_index)
            push!(values, distance)
        end
    end
end

function is_valid_tsp_solution(route::Vector{Int}, num_cities::Int)
    return has_correct_length(route, num_cities) && contains_all_cities(route, num_cities)
end

function has_correct_length(route::Vector{Int}, expected_length::Int)
    return length(route) == expected_length
end

function contains_all_cities(route::Vector{Int}, num_cities::Int)
    return Set(route) == Set(1:num_cities)
end

end # fim do módulo
