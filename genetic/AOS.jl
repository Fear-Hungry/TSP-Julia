module AOSModule

export UCB1Manager, select_operator!, update_stats!

using Random

"""
    UCB1Manager

Gerencia o estado e a lógica da Seleção Adaptativa de Operadores usando o algoritmo UCB1.
"""
mutable struct UCB1Manager
    operator_names::Vector{Symbol}
    num_operators::Int

    # Estatísticas
    n_ops::Vector{Int}         # Contador de quantas vezes cada operador foi usado
    s_ops::Vector{Float32}   # Soma das recompensas de cada operador
    total_applications::Int    # Total de vezes que qualquer operador foi chamado (t)

    # Hiperparâmetro
    exploration_constant::Float64 # O fator 'c' da fórmula UCB1

    function UCB1Manager(operator_names::Vector{Symbol}; exploration_constant::Float64 = 2.0)
        num_operators = length(operator_names)
        n_ops = zeros(Int, num_operators)
        s_ops = zeros(Float32, num_operators)
        new(operator_names, num_operators, n_ops, s_ops, 0, exploration_constant)
    end
end

"""
    select_operator!(manager::UCB1Manager) -> Symbol

Seleciona o melhor operador a ser usado com base nas pontuações UCB1.
"""
function select_operator!(manager::UCB1Manager)
    manager.total_applications += 1
    t = manager.total_applications
    ucb_scores = zeros(Float32, manager.num_operators)

    for i in 1:manager.num_operators
        ni = manager.n_ops[i]

        if ni == 0
            # Se um operador nunca foi usado, sua prioridade é infinita para garantir que seja explorado.
            ucb_scores[i] = Inf
        else
            # Média da recompensa (exploração)
            avg_reward = manager.s_ops[i] / ni

            # Termo de confiança (explotação)
            confidence_term = manager.exploration_constant * sqrt(log(t) / ni)

            ucb_scores[i] = avg_reward + confidence_term
        end
    end

    # Retorna o nome do operador com a maior pontuação UCB1
    best_op_idx = argmax(ucb_scores)
    return manager.operator_names[best_op_idx]
end

"""
    update_stats!(manager::UCB1Manager, op_name::Symbol, reward::Float32)

Atualiza as estatísticas (contador e soma de recompensas) para um operador após seu uso.
"""
function update_stats!(manager::UCB1Manager, op_name::Symbol, reward::Float32)
    op_idx = findfirst(==(op_name), manager.operator_names)
    if op_idx !== nothing
        manager.n_ops[op_idx] += 1
        # Garantimos que apenas recompensas positivas (melhorias) sejam somadas.
        manager.s_ops[op_idx] += max(0.0f0, reward)
    else
        @warn "Tentativa de atualizar estatísticas para um operador desconhecido: $op_name"
    end
end

end # fim do módulo
