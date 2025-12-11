"""
Sistema de Alocação de Membros em Squads - VERSÃO EXPANDIDA
===========================================================
Com visualizações e dados que forçam redistribuições visíveis.

Instalação: pip install ortools pandas matplotlib seaborn numpy
"""

from ortools.sat.python import cp_model
from dataclasses import dataclass, field
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict


# =============================================================================
# 1. ESTRUTURAS DE DADOS
# =============================================================================

@dataclass
class Pessoa:
    id: str
    nome: str
    papel: str
    senioridade: str
    skills: dict
    squad_atual: str = None
    linha_atual: str = None


@dataclass
class Squad:
    id: str
    nome: str
    linha: str
    capacidade_max: int
    capacidade_min: int
    necessidades: dict
    skills_desejadas: dict


@dataclass
class Restricao:
    tipo: str
    pessoa_id: str = None
    squad_id: str = None
    linha_id: str = None
    peso: int = 1


# =============================================================================
# 2. DADOS EXPANDIDOS (com pessoas MAL ALOCADAS propositalmente)
# =============================================================================

def criar_dados_expandidos():
    """
    Cria dados com pessoas claramente mal alocadas para forçar redistribuições.
    """
    
    pessoas = [
        # ===================== LINHA ALPHA =====================
        
        # --- Squad A1: Pagamentos (precisa: Python, AWS, Java backend) ---
        # BEM alocados:
        Pessoa("p1", "Ana Silva", "dev", "senior", 
               {"python": 5, "java": 5, "aws": 5, "docker": 4}, 
               "squad_a1", "alpha"),
        Pessoa("p2", "Bruno Costa", "dev", "pleno", 
               {"python": 4, "java": 4, "aws": 3, "sql": 4}, 
               "squad_a1", "alpha"),
        # MAL alocado - tem skills de DATA, não backend:
        Pessoa("p3", "Carlos Mendes", "data_engineer", "senior", 
               {"spark": 5, "python": 4, "kafka": 5, "airflow": 4, "java": 1}, 
               "squad_a1", "alpha"),  # Deveria estar em Analytics ou Data Platform
        # MAL alocado - tem skills de FRONTEND, não backend:
        Pessoa("p4", "Daniela Rocha", "dev", "pleno", 
               {"react": 5, "typescript": 5, "css": 4, "python": 2, "aws": 1}, 
               "squad_a1", "alpha"),  # Deveria estar em Checkout ou Mobile
        
        # --- Squad A2: Checkout (precisa: React, TypeScript, Node) ---
        # BEM alocado:
        Pessoa("p5", "Eduardo Lima", "dev", "senior", 
               {"react": 5, "typescript": 5, "node": 4, "graphql": 4}, 
               "squad_a2", "alpha"),
        # MAL alocado - tem skills de BACKEND Java, não frontend:
        Pessoa("p6", "Fernanda Dias", "dev", "pleno", 
               {"java": 5, "spring": 5, "aws": 4, "docker": 4, "react": 1}, 
               "squad_a2", "alpha"),  # Deveria estar em Pagamentos
        # MAL alocado - QA com skills de automação backend, squad precisa de frontend QA:
        Pessoa("p7", "Gabriel Santos", "qa", "senior", 
               {"selenium": 5, "java": 4, "api_testing": 5, "cypress": 1}, 
               "squad_a2", "alpha"),
        # BEM alocado:
        Pessoa("p8", "Helena Martins", "dev", "junior", 
               {"react": 4, "typescript": 3, "css": 4, "html": 5}, 
               "squad_a2", "alpha"),
        
        # --- Squad A3: Analytics (precisa: Python, Spark, ML, SQL) ---
        # BEM alocado:
        Pessoa("p9", "Igor Ferreira", "data_scientist", "senior", 
               {"python": 5, "ml": 5, "spark": 4, "sql": 5, "tensorflow": 4}, 
               "squad_a3", "alpha"),
        # MAL alocado - Dev frontend em squad de dados:
        Pessoa("p10", "Julia Alves", "dev", "pleno", 
               {"react": 5, "vue": 4, "typescript": 5, "node": 3, "python": 2}, 
               "squad_a3", "alpha"),  # Deveria estar em Checkout
        # MAL alocado - QA sem skills de dados:
        Pessoa("p11", "Kevin Souza", "qa", "pleno", 
               {"cypress": 5, "javascript": 4, "playwright": 4, "python": 2}, 
               "squad_a3", "alpha"),  # Deveria estar em Checkout
        
        # --- Squad A4: API Gateway (precisa: Java, Spring, AWS, Kubernetes) ---
        # BEM alocado:
        Pessoa("p12", "Larissa Nunes", "dev", "senior", 
               {"java": 5, "spring": 5, "kubernetes": 5, "aws": 5}, 
               "squad_a4", "alpha"),
        # BEM alocado:
        Pessoa("p13", "Marcos Oliveira", "dev", "pleno", 
               {"java": 4, "spring": 4, "docker": 4, "aws": 3}, 
               "squad_a4", "alpha"),
        # MAL alocado - Data Scientist em squad de infra:
        Pessoa("p14", "Natalia Campos", "data_scientist", "pleno", 
               {"python": 5, "ml": 4, "pandas": 5, "sql": 4, "java": 1}, 
               "squad_a4", "alpha"),  # Deveria estar em Analytics
        # MAL alocado - Dev Python/Data em squad Java:
        Pessoa("p15", "Oscar Ribeiro", "dev", "senior", 
               {"python": 5, "spark": 4, "sql": 5, "aws": 4, "java": 2}, 
               "squad_a4", "alpha"),  # Deveria estar em Analytics ou Data
        
        # ===================== PESSOAS DESALOCADAS =====================
        Pessoa("p16", "Patricia Lopes", "dev", "pleno", 
               {"react": 5, "typescript": 5, "next": 4, "css": 4}, 
               None, None),
        Pessoa("p17", "Quirino Melo", "data_engineer", "senior", 
               {"spark": 5, "kafka": 5, "python": 5, "airflow": 5, "aws": 4}, 
               None, None),
        Pessoa("p18", "Renata Gomes", "qa", "pleno", 
               {"cypress": 5, "playwright": 4, "typescript": 4, "api_testing": 3}, 
               None, None),
        Pessoa("p19", "Samuel Castro", "dev", "junior", 
               {"python": 3, "java": 3, "sql": 3, "git": 4}, 
               None, None),
        Pessoa("p20", "Tatiana Reis", "data_scientist", "senior", 
               {"python": 5, "ml": 5, "deep_learning": 5, "spark": 4, "sql": 5}, 
               None, None),
        Pessoa("p21", "Ulisses Braga", "dev", "pleno", 
               {"java": 5, "spring": 4, "aws": 4, "kubernetes": 3}, 
               None, None),
    ]
    
    squads = [
        # Linha Alpha
        Squad("squad_a1", "Pagamentos", "alpha", 
              capacidade_max=5, capacidade_min=3,
              necessidades={"dev": 3, "qa": 1},
              skills_desejadas={"python": 4, "java": 4, "aws": 4}),
        
        Squad("squad_a2", "Checkout", "alpha", 
              capacidade_max=5, capacidade_min=3,
              necessidades={"dev": 3, "qa": 1},
              skills_desejadas={"react": 4, "typescript": 4, "node": 3}),
        
        Squad("squad_a3", "Analytics", "alpha", 
              capacidade_max=5, capacidade_min=2,
              necessidades={"data_scientist": 2, "data_engineer": 1, "dev": 1},
              skills_desejadas={"python": 4, "spark": 4, "ml": 3, "sql": 4}),
        
        Squad("squad_a4", "API Gateway", "alpha", 
              capacidade_max=4, capacidade_min=2,
              necessidades={"dev": 3, "qa": 1},
              skills_desejadas={"java": 5, "spring": 4, "kubernetes": 4, "aws": 4}),
        
        # Linha Beta (para desalocados)
        Squad("squad_b1", "Mobile", "beta", 
              capacidade_max=5, capacidade_min=3,
              necessidades={"dev": 3, "qa": 1},
              skills_desejadas={"react": 4, "typescript": 4, "react_native": 3}),
        
        Squad("squad_b2", "Data Platform", "beta", 
              capacidade_max=5, capacidade_min=2,
              necessidades={"data_engineer": 2, "data_scientist": 1, "dev": 1},
              skills_desejadas={"spark": 5, "kafka": 4, "python": 4, "airflow": 4}),
    ]
    
    restricoes = [
        # Ana deve permanecer em Pagamentos (tech lead)
        Restricao("forcar", pessoa_id="p1", squad_id="squad_a1"),
    ]
    
    return pessoas, squads, restricoes


# =============================================================================
# 3. FUNÇÕES DE SCORING
# =============================================================================

def calcular_fit_skill(pessoa: Pessoa, squad: Squad) -> float:
    """Calcula score de fit baseado nas skills (0-100)"""
    if not squad.skills_desejadas:
        return 50.0
    
    score = 0
    max_score = 0
    
    for skill, nivel_minimo in squad.skills_desejadas.items():
        max_score += 5
        nivel_pessoa = pessoa.skills.get(skill, 0)
        
        if nivel_pessoa >= nivel_minimo:
            score += min(nivel_pessoa, 5)
        else:
            score += nivel_pessoa * 0.3  # Penalidade maior
    
    return (score / max_score) * 100 if max_score > 0 else 50.0


def calcular_fit_papel(pessoa: Pessoa, squad: Squad) -> float:
    """Calcula score baseado no papel (0-100)"""
    if pessoa.papel in squad.necessidades:
        return 100.0
    return 20.0


def calcular_fit_total(pessoa: Pessoa, squad: Squad) -> float:
    """Combina fatores de fit"""
    return calcular_fit_skill(pessoa, squad) * 0.7 + calcular_fit_papel(pessoa, squad) * 0.3


def criar_matriz_fit(pessoas: list, squads: list) -> pd.DataFrame:
    """Cria matriz de fit pessoas x squads"""
    data = []
    for p in pessoas:
        row = {"pessoa": p.nome, "pessoa_id": p.id, "papel": p.papel}
        for s in squads:
            row[s.nome] = round(calcular_fit_total(p, s), 1)
        data.append(row)
    return pd.DataFrame(data)


# =============================================================================
# 4. SOLVER - REDISTRIBUIÇÃO
# =============================================================================

def redistribuir_linha(pessoas: list, squads: list, linha: str, 
                       restricoes: list = None, bonus_permanencia: int = 5) -> dict:
    """Redistribui membros entre squads de uma linha"""
    
    pessoas_linha = [p for p in pessoas if p.linha_atual == linha]
    squads_linha = [s for s in squads if s.linha == linha]
    
    if not pessoas_linha or not squads_linha:
        return {}, {}
    
    # Guardar alocação anterior
    alocacao_anterior = {p.id: p.squad_atual for p in pessoas_linha}
    
    model = cp_model.CpModel()
    
    # Variáveis
    x = {}
    for p in pessoas_linha:
        for s in squads_linha:
            x[p.id, s.id] = model.NewBoolVar(f'x_{p.id}_{s.id}')
    
    # Restrições básicas
    for p in pessoas_linha:
        model.Add(sum(x[p.id, s.id] for s in squads_linha) == 1)
    
    for s in squads_linha:
        membros = sum(x[p.id, s.id] for p in pessoas_linha)
        model.Add(membros <= s.capacidade_max)
        model.Add(membros >= s.capacidade_min)
    
    # Restrições parametrizáveis
    if restricoes:
        for r in restricoes:
            if r.tipo == "bloquear" and r.pessoa_id and r.squad_id:
                if (r.pessoa_id, r.squad_id) in x:
                    model.Add(x[r.pessoa_id, r.squad_id] == 0)
            elif r.tipo == "forcar" and r.pessoa_id and r.squad_id:
                if (r.pessoa_id, r.squad_id) in x:
                    model.Add(x[r.pessoa_id, r.squad_id] == 1)
    
    # Objetivo
    objetivo = []
    for p in pessoas_linha:
        for s in squads_linha:
            fit = int(calcular_fit_total(p, s) * 10)  # Escalar para inteiro
            if p.squad_atual == s.id:
                fit += bonus_permanencia
            objetivo.append(fit * x[p.id, s.id])
    
    model.Maximize(sum(objetivo))
    
    # Resolver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    status = solver.Solve(model)
    
    # Extrair resultados
    resultado = {}
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        for p in pessoas_linha:
            for s in squads_linha:
                if solver.Value(x[p.id, s.id]) == 1:
                    resultado[p.id] = s.id
    
    return resultado, alocacao_anterior


def alocar_desalocados(pessoas: list, squads: list, restricoes: list = None) -> dict:
    """Aloca pessoas desalocadas nas squads com déficit"""
    
    desalocados = [p for p in pessoas if p.squad_atual is None]
    if not desalocados:
        return {}
    
    # Calcular vagas
    alocados_por_squad = defaultdict(int)
    for p in pessoas:
        if p.squad_atual:
            alocados_por_squad[p.squad_atual] += 1
    
    squads_com_vaga = []
    for s in squads:
        vagas = s.capacidade_max - alocados_por_squad.get(s.id, 0)
        if vagas > 0:
            squads_com_vaga.append((s, vagas))
    
    if not squads_com_vaga:
        return {}
    
    model = cp_model.CpModel()
    
    x = {}
    for p in desalocados:
        for s, _ in squads_com_vaga:
            x[p.id, s.id] = model.NewBoolVar(f'x_{p.id}_{s.id}')
    
    for p in desalocados:
        model.Add(sum(x[p.id, s.id] for s, _ in squads_com_vaga) <= 1)
    
    for s, vagas in squads_com_vaga:
        model.Add(sum(x[p.id, s.id] for p in desalocados) <= vagas)
    
    if restricoes:
        for r in restricoes:
            if r.tipo == "forcar" and r.pessoa_id and r.squad_id:
                if (r.pessoa_id, r.squad_id) in x:
                    model.Add(x[r.pessoa_id, r.squad_id] == 1)
    
    objetivo = []
    for p in desalocados:
        for s, _ in squads_com_vaga:
            fit = int(calcular_fit_total(p, s) * 10) + 500  # Bonus alocação
            objetivo.append(fit * x[p.id, s.id])
    
    model.Maximize(sum(objetivo))
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    status = solver.Solve(model)
    
    resultado = {}
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        for p in desalocados:
            for s, _ in squads_com_vaga:
                if solver.Value(x[p.id, s.id]) == 1:
                    resultado[p.id] = s.id
    
    return resultado


# =============================================================================
# 5. VISUALIZAÇÕES
# =============================================================================

def plot_heatmap_fit(pessoas: list, squads: list, titulo: str = "Matriz de Fit"):
    """Heatmap de fit pessoas x squads"""
    
    df = criar_matriz_fit(pessoas, squads)
    
    # Preparar dados para heatmap
    squad_names = [s.nome for s in squads]
    heatmap_data = df[squad_names].values
    pessoas_names = df['pessoa'].tolist()
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(pessoas) * 0.4)))
    
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt='.0f',
        cmap='RdYlGn',
        xticklabels=squad_names,
        yticklabels=pessoas_names,
        vmin=0, 
        vmax=100,
        ax=ax,
        cbar_kws={'label': 'Fit Score'}
    )
    
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.set_xlabel('Squads', fontsize=11)
    ax.set_ylabel('Pessoas', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig


def plot_mudancas_sankey(pessoas: list, squads: list, 
                         alocacao_anterior: dict, alocacao_nova: dict):
    """Visualiza fluxo de mudanças entre squads (versão simplificada com barras)"""
    
    squad_map = {s.id: s.nome for s in squads}
    
    # Contar mudanças
    mudancas = []
    permanencias = []
    
    for pessoa_id, squad_nova in alocacao_nova.items():
        squad_antiga = alocacao_anterior.get(pessoa_id)
        pessoa = next((p for p in pessoas if p.id == pessoa_id), None)
        
        if pessoa and squad_antiga and squad_antiga != squad_nova:
            mudancas.append({
                'pessoa': pessoa.nome,
                'de': squad_map.get(squad_antiga, squad_antiga),
                'para': squad_map.get(squad_nova, squad_nova),
                'papel': pessoa.papel
            })
        elif pessoa and squad_antiga == squad_nova:
            permanencias.append({
                'pessoa': pessoa.nome,
                'squad': squad_map.get(squad_nova, squad_nova),
                'papel': pessoa.papel
            })
    
    if not mudancas:
        print("Nenhuma mudança para visualizar.")
        return None
    
    # Criar figura com subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Gráfico 1: Mudanças por Squad ---
    df_mud = pd.DataFrame(mudancas)
    
    # Contar saídas e entradas por squad
    saidas = df_mud.groupby('de').size()
    entradas = df_mud.groupby('para').size()
    
    all_squads = list(set(saidas.index) | set(entradas.index))
    
    x = np.arange(len(all_squads))
    width = 0.35
    
    saidas_vals = [saidas.get(s, 0) for s in all_squads]
    entradas_vals = [entradas.get(s, 0) for s in all_squads]
    
    bars1 = axes[0].bar(x - width/2, saidas_vals, width, label='Saídas', color='#e74c3c')
    bars2 = axes[0].bar(x + width/2, entradas_vals, width, label='Entradas', color='#27ae60')
    
    axes[0].set_ylabel('Número de Pessoas')
    axes[0].set_title('Fluxo de Pessoas por Squad', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(all_squads, rotation=45, ha='right')
    axes[0].legend()
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Adicionar valores nas barras
    for bar in bars1:
        if bar.get_height() > 0:
            axes[0].annotate(f'{int(bar.get_height())}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        if bar.get_height() > 0:
            axes[0].annotate(f'{int(bar.get_height())}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           ha='center', va='bottom', fontsize=10)
    
    # --- Gráfico 2: Lista de Mudanças ---
    axes[1].axis('off')
    
    table_data = [[m['pessoa'], m['papel'], m['de'], '→', m['para']] for m in mudancas]
    
    table = axes[1].table(
        cellText=table_data,
        colLabels=['Pessoa', 'Papel', 'De', '', 'Para'],
        loc='center',
        cellLoc='center',
        colWidths=[0.25, 0.15, 0.25, 0.05, 0.25]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Colorir header
    for i in range(5):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    axes[1].set_title('Detalhamento das Mudanças', fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig


def plot_comparacao_antes_depois(pessoas: list, squads: list,
                                  alocacao_anterior: dict, alocacao_nova: dict):
    """Compara fit médio antes e depois por squad"""
    
    squads_linha = [s for s in squads if s.linha == "alpha"]
    squad_map = {s.id: s for s in squads_linha}
    
    # Calcular fit médio antes e depois
    fit_antes = defaultdict(list)
    fit_depois = defaultdict(list)
    
    for pessoa_id, squad_antes_id in alocacao_anterior.items():
        pessoa = next((p for p in pessoas if p.id == pessoa_id), None)
        if pessoa and squad_antes_id in squad_map:
            fit = calcular_fit_total(pessoa, squad_map[squad_antes_id])
            fit_antes[squad_map[squad_antes_id].nome].append(fit)
    
    for pessoa_id, squad_depois_id in alocacao_nova.items():
        pessoa = next((p for p in pessoas if p.id == pessoa_id), None)
        if pessoa and squad_depois_id in squad_map:
            fit = calcular_fit_total(pessoa, squad_map[squad_depois_id])
            fit_depois[squad_map[squad_depois_id].nome].append(fit)
    
    # Preparar dados
    squad_names = [s.nome for s in squads_linha]
    media_antes = [np.mean(fit_antes.get(s, [0])) for s in squad_names]
    media_depois = [np.mean(fit_depois.get(s, [0])) for s in squad_names]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(squad_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, media_antes, width, label='Antes', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, media_depois, width, label='Depois', color='#27ae60', alpha=0.8)
    
    ax.set_ylabel('Fit Médio', fontsize=11)
    ax.set_title('Comparação de Fit Médio por Squad: Antes vs Depois', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(squad_names, fontsize=10)
    ax.legend()
    ax.set_ylim(0, 100)
    
    # Linha de referência
    ax.axhline(y=70, color='gray', linestyle='--', alpha=0.5, label='Meta (70)')
    
    # Adicionar valores e deltas
    for i, (antes, depois) in enumerate(zip(media_antes, media_depois)):
        delta = depois - antes
        color = '#27ae60' if delta > 0 else '#e74c3c'
        ax.annotate(f'{antes:.0f}', xy=(x[i] - width/2, antes + 1), ha='center', fontsize=9)
        ax.annotate(f'{depois:.0f}', xy=(x[i] + width/2, depois + 1), ha='center', fontsize=9)
        
        if abs(delta) > 0.1:
            ax.annotate(f'{"+" if delta > 0 else ""}{delta:.1f}',
                       xy=(x[i], max(antes, depois) + 5),
                       ha='center', fontsize=10, fontweight='bold', color=color)
    
    plt.tight_layout()
    return fig


def plot_distribuicao_skills_squad(pessoas: list, squads: list, alocacao: dict):
    """Radar chart de skills por squad"""
    
    squads_alpha = [s for s in squads if s.linha == "alpha"]
    
    # Coletar todas as skills únicas
    all_skills = set()
    for s in squads_alpha:
        all_skills.update(s.skills_desejadas.keys())
    all_skills = sorted(list(all_skills))
    
    if len(all_skills) < 3:
        print("Skills insuficientes para radar chart")
        return None
    
    # Limitar a 8 skills para visualização
    all_skills = all_skills[:8]
    
    n_skills = len(all_skills)
    angles = np.linspace(0, 2 * np.pi, n_skills, endpoint=False).tolist()
    angles += angles[:1]  # Fechar o círculo
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), subplot_kw=dict(polar=True))
    axes = axes.flatten()
    
    colors = ['#3498db', '#e74c3c', '#27ae60', '#9b59b6']
    
    for idx, squad in enumerate(squads_alpha[:4]):
        ax = axes[idx]
        
        # Encontrar pessoas nesta squad
        pessoas_squad = [p for p in pessoas if alocacao.get(p.id) == squad.id]
        
        if not pessoas_squad:
            continue
        
        # Calcular média de skills
        skill_means = []
        for skill in all_skills:
            valores = [p.skills.get(skill, 0) for p in pessoas_squad]
            skill_means.append(np.mean(valores))
        
        skill_means += skill_means[:1]
        
        # Skills desejadas pela squad
        skill_required = [squad.skills_desejadas.get(skill, 0) for skill in all_skills]
        skill_required += skill_required[:1]
        
        # Plot
        ax.plot(angles, skill_means, 'o-', linewidth=2, color=colors[idx], label='Atual')
        ax.fill(angles, skill_means, alpha=0.25, color=colors[idx])
        ax.plot(angles, skill_required, '--', linewidth=2, color='gray', alpha=0.7, label='Desejado')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(all_skills, size=9)
        ax.set_ylim(0, 5)
        ax.set_title(f'{squad.nome}\n({len(pessoas_squad)} pessoas)', size=11, fontweight='bold', pad=10)
        ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('Perfil de Skills por Squad (Atual vs Desejado)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_resumo_alocacao(pessoas: list, squads: list, alocacao: dict):
    """Dashboard resumo da alocação"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # --- 1. Ocupação por Squad ---
    ax1 = axes[0, 0]
    squad_names = [s.nome for s in squads]
    ocupacao = []
    capacidade = []
    
    for s in squads:
        count = sum(1 for pid, sid in alocacao.items() if sid == s.id)
        count += sum(1 for p in pessoas if p.squad_atual == s.id and p.id not in alocacao)
        ocupacao.append(count)
        capacidade.append(s.capacidade_max)
    
    x = np.arange(len(squad_names))
    bars = ax1.bar(x, ocupacao, color='#3498db', alpha=0.8)
    ax1.bar(x, capacidade, fill=False, edgecolor='gray', linestyle='--', linewidth=2)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(squad_names, rotation=45, ha='right')
    ax1.set_ylabel('Pessoas')
    ax1.set_title('Ocupação vs Capacidade', fontweight='bold')
    
    for i, (o, c) in enumerate(zip(ocupacao, capacidade)):
        color = '#27ae60' if o >= squads[i].capacidade_min else '#e74c3c'
        ax1.annotate(f'{o}/{c}', xy=(i, o + 0.2), ha='center', fontsize=10, color=color)
    
    # --- 2. Distribuição por Papel ---
    ax2 = axes[0, 1]
    papeis = defaultdict(int)
    for p in pessoas:
        papeis[p.papel] += 1
    
    colors_papel = ['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6']
    ax2.pie(papeis.values(), labels=papeis.keys(), autopct='%1.0f%%', 
            colors=colors_papel[:len(papeis)], startangle=90)
    ax2.set_title('Distribuição por Papel', fontweight='bold')
    
    # --- 3. Fit Médio por Squad ---
    ax3 = axes[1, 0]
    fit_medio = []
    
    for s in squads:
        fits = []
        for p in pessoas:
            if alocacao.get(p.id) == s.id or (p.squad_atual == s.id and p.id not in alocacao):
                fits.append(calcular_fit_total(p, s))
        fit_medio.append(np.mean(fits) if fits else 0)
    
    colors_fit = ['#27ae60' if f >= 70 else '#f39c12' if f >= 50 else '#e74c3c' for f in fit_medio]
    bars = ax3.barh(squad_names, fit_medio, color=colors_fit, alpha=0.8)
    ax3.set_xlim(0, 100)
    ax3.axvline(x=70, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Fit Médio')
    ax3.set_title('Fit Médio por Squad', fontweight='bold')
    
    for i, v in enumerate(fit_medio):
        ax3.annotate(f'{v:.0f}', xy=(v + 1, i), va='center', fontsize=10)
    
    # --- 4. Status Geral ---
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    total_pessoas = len(pessoas)
    alocados = len([p for p in pessoas if p.squad_atual or p.id in alocacao])
    desalocados = total_pessoas - alocados
    fit_geral = np.mean([f for f in fit_medio if f > 0])
    
    stats_text = f"""
    RESUMO GERAL
    ════════════════════════════
    
    Total de Pessoas:     {total_pessoas}
    Alocados:             {alocados}
    Desalocados:          {desalocados}
    
    Fit Médio Geral:      {fit_geral:.1f}
    
    Squads:               {len(squads)}
    Linhas:               {len(set(s.linha for s in squads))}
    """
    
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
             fontfamily='monospace', verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
    
    plt.suptitle('Dashboard de Alocação', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# 6. RELATÓRIO TEXTUAL
# =============================================================================

def imprimir_relatorio(pessoas: list, squads: list, 
                       alocacao_anterior: dict, alocacao_nova: dict,
                       titulo: str):
    """Imprime relatório detalhado das mudanças"""
    
    print(f"\n{'='*70}")
    print(f" {titulo}")
    print(f"{'='*70}")
    
    squad_map = {s.id: s.nome for s in squads}
    pessoa_map = {p.id: p for p in pessoas}
    
    mudancas = []
    permanencias = []
    
    for pessoa_id, squad_nova in alocacao_nova.items():
        squad_antiga = alocacao_anterior.get(pessoa_id)
        pessoa = pessoa_map.get(pessoa_id)
        
        if not pessoa:
            continue
            
        squad_obj = next((s for s in squads if s.id == squad_nova), None)
        fit_novo = calcular_fit_total(pessoa, squad_obj) if squad_obj else 0
        
        if squad_antiga and squad_antiga != squad_nova:
            squad_antiga_obj = next((s for s in squads if s.id == squad_antiga), None)
            fit_antigo = calcular_fit_total(pessoa, squad_antiga_obj) if squad_antiga_obj else 0
            
            mudancas.append({
                'pessoa': pessoa,
                'de': squad_map.get(squad_antiga, squad_antiga),
                'para': squad_map.get(squad_nova, squad_nova),
                'fit_antigo': fit_antigo,
                'fit_novo': fit_novo,
                'delta': fit_novo - fit_antigo
            })
        else:
            permanencias.append({
                'pessoa': pessoa,
                'squad': squad_map.get(squad_nova, squad_nova),
                'fit': fit_novo
            })
    
    # Mudanças
    if mudancas:
        print(f"\n🔄 MUDANÇAS ({len(mudancas)} pessoas):")
        print("-" * 70)
        print(f"{'Pessoa':<20} {'Papel':<12} {'De':<15} {'Para':<15} {'Δ Fit':>8}")
        print("-" * 70)
        
        for m in sorted(mudancas, key=lambda x: -x['delta']):
            delta_str = f"+{m['delta']:.0f}" if m['delta'] > 0 else f"{m['delta']:.0f}"
            delta_emoji = "📈" if m['delta'] > 0 else "📉"
            print(f"{m['pessoa'].nome:<20} {m['pessoa'].papel:<12} {m['de']:<15} {m['para']:<15} {delta_str:>6} {delta_emoji}")
    
    # Permanências
    if permanencias:
        print(f"\n✅ PERMANÊNCIAS ({len(permanencias)} pessoas):")
        print("-" * 50)
        for p in permanencias:
            print(f"   {p['pessoa'].nome:<20} → {p['squad']:<15} (fit: {p['fit']:.0f})")
    
    # Estatísticas
    if mudancas:
        delta_medio = np.mean([m['delta'] for m in mudancas])
        melhorias = len([m for m in mudancas if m['delta'] > 0])
        print(f"\n📊 ESTATÍSTICAS:")
        print(f"   Δ Fit médio nas mudanças: {delta_medio:+.1f}")
        print(f"   Mudanças com melhoria: {melhorias}/{len(mudancas)}")


# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    print("Carregando dados...")
    pessoas, squads, restricoes = criar_dados_expandidos()
    
    print(f"Total: {len(pessoas)} pessoas, {len(squads)} squads")
    
    # ========== PROBLEMA 1: Redistribuição da Linha Alpha ==========
    print("\n" + "="*70)
    print(" EXECUTANDO REDISTRIBUIÇÃO - LINHA ALPHA")
    print("="*70)
    
    alocacao_nova, alocacao_anterior = redistribuir_linha(
        pessoas, squads, "alpha", restricoes, bonus_permanencia=3
    )
    
    imprimir_relatorio(
        pessoas, squads, alocacao_anterior, alocacao_nova,
        "REDISTRIBUIÇÃO - LINHA ALPHA"
    )
    
    # ========== PROBLEMA 2: Alocação de Desalocados ==========
    print("\n" + "="*70)
    print(" EXECUTANDO ALOCAÇÃO DE DESALOCADOS")
    print("="*70)
    
    alocacao_desalocados = alocar_desalocados(pessoas, squads, restricoes)
    
    desalocados_anterior = {p.id: None for p in pessoas if p.squad_atual is None}
    
    imprimir_relatorio(
        pessoas, squads, desalocados_anterior, alocacao_desalocados,
        "ALOCAÇÃO DE DESALOCADOS"
    )
    
    # ========== VISUALIZAÇÕES ==========
    print("\n" + "="*70)
    print(" GERANDO VISUALIZAÇÕES")
    print("="*70)
    
    # Combinar alocações para visualização completa
    alocacao_completa = {**alocacao_nova, **alocacao_desalocados}
    
    # 1. Heatmap de Fit
    print("  → Gerando heatmap de fit...")
    fig1 = plot_heatmap_fit(
        [p for p in pessoas if p.linha_atual == "alpha"], 
        [s for s in squads if s.linha == "alpha"],
        "Matriz de Fit - Linha Alpha"
    )
    fig1.savefig('/home/claude/01_heatmap_fit.png', dpi=150, bbox_inches='tight')
    print("    Salvo: 01_heatmap_fit.png")
    
    # 2. Mudanças
    print("  → Gerando visualização de mudanças...")
    fig2 = plot_mudancas_sankey(pessoas, squads, alocacao_anterior, alocacao_nova)
    if fig2:
        fig2.savefig('/home/claude/02_mudancas.png', dpi=150, bbox_inches='tight')
        print("    Salvo: 02_mudancas.png")
    
    # 3. Comparação Antes/Depois
    print("  → Gerando comparação antes/depois...")
    fig3 = plot_comparacao_antes_depois(pessoas, squads, alocacao_anterior, alocacao_nova)
    fig3.savefig('/home/claude/03_comparacao_antes_depois.png', dpi=150, bbox_inches='tight')
    print("    Salvo: 03_comparacao_antes_depois.png")
    
    # 4. Radar de Skills
    print("  → Gerando radar de skills...")
    fig4 = plot_distribuicao_skills_squad(pessoas, squads, alocacao_nova)
    if fig4:
        fig4.savefig('/home/claude/04_radar_skills.png', dpi=150, bbox_inches='tight')
        print("    Salvo: 04_radar_skills.png")
    
    # 5. Dashboard Resumo
    print("  → Gerando dashboard resumo...")
    fig5 = plot_resumo_alocacao(pessoas, squads, alocacao_completa)
    fig5.savefig('/home/claude/05_dashboard_resumo.png', dpi=150, bbox_inches='tight')
    print("    Salvo: 05_dashboard_resumo.png")
    
    plt.close('all')
    
    print("\n" + "="*70)
    print(" CONCLUÍDO!")
    print("="*70)
    print("\nArquivos gerados:")
    print("  • 01_heatmap_fit.png")
    print("  • 02_mudancas.png")
    print("  • 03_comparacao_antes_depois.png")
    print("  • 04_radar_skills.png")
    print("  • 05_dashboard_resumo.png")


if __name__ == "__main__":
    main()

