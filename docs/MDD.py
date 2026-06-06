import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

def run_mdd():
    np.random.seed(42)
    plt.figure(figsize=(10, 6))
    existing_system_responses = np.random.normal(loc=3.5, scale=0.4, size=500000)
    improved_system_responses = np.random.normal(loc=2.0, scale=0.4, size=500000)
    
    sns.kdeplot(
        existing_system_responses, label="Существующая система", fill=True, color="red"
    )
    sns.kdeplot(
        improved_system_responses, label="Улучшенная система", fill=True, color="green"
    )
    
    plt.title("Сравнение времени отклика системы")
    plt.xlabel("Время отклика (секунды)")
    plt.ylabel("Наблюдения")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    num_ticks = 5
    new_yticks = np.linspace(ymin, ymax, num_ticks)
    new_yticklabels = [
        f"{int((tick / ymax) * 100)}%" if ymax != 0 else "0%" for tick in new_yticks
    ]
    ax.set_yticks(new_yticks)
    ax.set_yticklabels(new_yticklabels)
     
    output_filename = "mdd_comparison.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"График распределений успешно сохранен в файл: {output_filename}")

    t_stat, p_value = stats.ttest_ind(
        existing_system_responses, 
        improved_system_responses, 
        alternative='greater'
    )

    print("Результаты стат.анализа:")
    print(f"T-statistic: {t_stat:.2f}")
    print(f"P-value: {p_value}")

    if p_value < 0.05:
        print("Улучшение статистически значимо.")
    else:
        print("Улучшение статистически не значимо.")

if __name__ == "__main__":
    run_mdd()