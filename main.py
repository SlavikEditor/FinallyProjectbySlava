"""Climate Watcher — Telegram Bot
Features: fetch NASA GISS (temp) & NOAA MLO (CO2), save CSV, analyze, plot, report
"""

import argparse, io, sys, os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from typing import TYPE_CHECKING

# Загружаем переменные из .env файла
load_dotenv()

try:
    import requests
except Exception:
    requests = None

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes
except Exception:
    print("Warning: python-telegram-bot not installed")

if TYPE_CHECKING:
    # Help static analyzers resolve types without importing at runtime
    import pandas as pd  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    from scipy import stats  # type: ignore
else:
    try:
        import pandas as pd
    except Exception:
        pd = None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None
    try:
        from scipy import stats
    except Exception:
        stats = None

# Единицы измерения для показателей
UNITS = {
    'Temperature': '°C (градусы Цельсия)',
    'CO2': 'ppm (части на миллион)'
}

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"; OUT_DIR = ROOT / "output"
DATA_DIR.mkdir(exist_ok=True); OUT_DIR.mkdir(exist_ok=True)


def append_or_update_csv(path: Path, df):
    if pd is None: raise RuntimeError('pandas required')
    if df.empty: return
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(subset=['year'], keep='last').sort_values('year')
    df.to_csv(path, index=False)


def fetch_nasa_global_temp():
    if requests is None or pd is None: raise RuntimeError('requests+pandas required')
    url = 'https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv'
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        txt=r.text
    except requests.exceptions.ConnectTimeout:
        raise RuntimeError('NASA сервер не отвечает (истекло время ожидания)')
    except requests.exceptions.ConnectionError:
        raise RuntimeError('Ошибка подключения к NASA серверу (проверьте интернет)')
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f'NASA сервер вернул ошибку {e.response.status_code}')
    except requests.exceptions.Timeout:
        raise RuntimeError('Истекло время ожидания ответа от NASA')
    except Exception as e:
        raise RuntimeError(f'Ошибка загрузки с NASA: {str(e)[:100]}')
    
    s = txt.find('Year'); df=pd.read_csv(io.StringIO(txt[s:]), na_values=['****'], skipinitialspace=True)
    
    # Попробуем найти подходящие колонки для года и значения
    year_col = None
    value_col = None
    
    for col in df.columns:
        if 'year' in col.lower(): year_col = col
        if 'annual' in col.lower() or col.strip() == 'Annual': value_col = col
    
    if year_col is None: year_col = df.columns[0]
    if value_col is None: value_col = df.columns[1] if len(df.columns) > 1 else None
    
    if value_col is None: raise RuntimeError('Не удалось найти данные в файле NASA')
    
    out = df[[year_col, value_col]].rename(columns={year_col:'year', value_col:'value'})
    out['value']=pd.to_numeric(out['value'],errors='coerce'); out=out.dropna(subset=['value']).astype({'year':int})
    out['source']='NASA-GISS'; out['fetched_at']=datetime.utcnow().isoformat(); return out[['year','value','source','fetched_at']]


def fetch_noaa_co2():
    if requests is None or pd is None: raise RuntimeError('requests+pandas required')
    url='https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt'
    try:
        r=requests.get(url,timeout=20)
        r.raise_for_status()
    except requests.exceptions.ConnectTimeout:
        raise RuntimeError('NOAA сервер не отвечает (истекло время ожидания)')
    except requests.exceptions.ConnectionError:
        raise RuntimeError('Ошибка подключения к NOAA серверу (проверьте интернет)')
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f'NOAA сервер вернул ошибку {e.response.status_code}')
    except requests.exceptions.Timeout:
        raise RuntimeError('Истекло время ожидания ответа от NOAA')
    except Exception as e:
        raise RuntimeError(f'Ошибка загрузки с NOAA: {str(e)[:100]}')
    
    rows=[]
    for ln in r.text.splitlines():
        if ln.startswith('#') or not ln.strip(): continue
        p=ln.split();
        if len(p)<4: continue
        rows.append((int(p[0]), float(p[3])))
    df=pd.DataFrame(rows,columns=['year','value']).groupby('year').mean().reset_index()
    df['source']='NOAA-MLO'; df['fetched_at']=datetime.utcnow().isoformat(); return df


def save_indicator(df,name): append_or_update_csv(DATA_DIR/f"{name}.csv", df)

def load_indicator(name):
    if pd is None:
        raise RuntimeError("pandas is required to load indicators. Install with: pip install pandas")
    p = DATA_DIR / f"{name}.csv"
    if not p.exists():
        return pd.DataFrame(columns=['year','value','source','fetched_at'])
    return pd.read_csv(p)


def analyze_trend(df, years=10):
    if stats is None: raise RuntimeError('scipy required')
    df=df.dropna(subset=['value']).sort_values('year'); last=int(df['year'].max()); sub=df[df['year']>=last-years+1]
    if len(sub)<2: return None
    x=sub['year'].values; y=sub['value'].values; s,_,r,_,_=stats.linregress(x,y)
    return {'years':years,'slope_per_year':float(s),'r2':float(r**2),'abs_change':float(y[-1]-y[0]),'pct_change':float((y[-1]-y[0])/abs(y[0])*100) if y[0]!=0 else None}


def plot_timeseries(df,title,ylabel, years=None):
    if plt is None: print('matplotlib missing; skipping plots'); return
    df=df.dropna(subset=['value']).sort_values('year')
    if df.empty: return
    
    # Фильтруем данные по периоду, если указан
    if years:
        last_year = int(df['year'].max())
        first_year = last_year - years + 1
        df = df[df['year'] >= first_year]
        title_with_period = f"{title} ({first_year}-{last_year})"
    else:
        title_with_period = title
    
    plt.figure(figsize=(8,3)); plt.plot(df['year'],df['value'],marker='o'); plt.title(title_with_period); plt.xlabel('Year'); plt.ylabel(ylabel); plt.grid(True)
    out=OUT_DIR/f"{title.replace(' ','_')}.png"; plt.tight_layout(); plt.savefig(out); plt.close(); print(f"Saved {out}")
    return out


def make_report(summary):
    lines=[]
    for k,v in summary.items():
        unit = UNITS.get(k, '')
        if v is None:
            lines.append(f"{k}: нет данных")
        else:
            lines.append(f"{k} ({unit}): за {v['years']}л изм {v['abs_change']:+.3f} (накл {v['slope_per_year']:+.4f}/год, R²={v['r2']:.2f})")
    
    # Добавляем легенду
    legend = "\n\n📋 *Легенда:*\n"
    legend += "• *изм* — абсолютное изменение значения за период\n"
    legend += "• *накл* — наклон (скорость изменения в год)\n"
    legend += "• *R²* — качество соответствия (1.0 = идеально, 0 = без связи)\n"
    legend += "• *年数* — количество лет для анализа\n"
    
    result = '\n'.join(lines) if lines else 'Нет данных'
    return result + legend


def create_sample_data():
    if pd is None: raise RuntimeError('pandas required')
    now = datetime.utcnow().year
    years = list(range(1980, now+1))
    temp = [0.1 + 0.02*(y-1980) for y in years]
    co2 = [340 + 1.5*(y-1980) for y in years]
    dfs = {
        'global_temp': pd.DataFrame({'year':years,'value':temp,'source':'SYNTHETIC','fetched_at':datetime.utcnow().isoformat()}),
        'co2': pd.DataFrame({'year':years,'value':co2,'source':'SYNTHETIC','fetched_at':datetime.utcnow().isoformat()}),
    }
    for k,df in dfs.items(): save_indicator(df, k)
    print('Sample data created')


def interactive_menu():
    """Интерактивное меню управления"""
    print("Climate Watcher — Панель управления")
    print("="*50)
    print("\n Доступные действия: ")
    print("  1. Создать пример данных (2025-2026)")
    print("  2. Обновить данные из интернета")
    print("  3. Анализировать тренд температуры и CO2")
    print("  4. Построить графики")
    print("  5. Показать отчет")
    print("  6. Все сразу (создать данные + анализ + графики + отчет)")
    print("  0. Выход")
    
    choice = input("\nВыберите действие (0-6): ").strip()
    return choice

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start"""
    msg = """🌍 *Climate Watcher Bot*

Доступные команды:
/sample — Создать пример данных
/update — Обновить данные из интернета
/analyze — Анализировать тренд
/plot — Построить графики
/report — Показать отчет
/all — Всё сразу (создать + анализ + графики + отчет)
/help — Справка"""
    await update.message.reply_text(msg, parse_mode="Markdown")

async def cmd_sample(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /sample - создать и показать пример данных"""
    try:
        create_sample_data()
        
        # Загружаем и показываем данные
        indicators={'global_temp':(fetch_nasa_global_temp,'Temperature'), 'co2':(fetch_noaa_co2,'CO2')}
        
        for k,(_,name) in indicators.items():
            try:
                df = load_indicator(k)
                unit = UNITS.get(name, '')
                
                msg = f"✓ *{name}* ({unit}):\n\n"
                msg += f"📊 Всего записей: {len(df)}\n"
                msg += f"Годы: {int(df['year'].min())} - {int(df['year'].max())}\n"
                msg += f"Значение: мин={df['value'].min():.2f}, макс={df['value'].max():.2f}\n\n"
                
                # Показываем первые и последние  5 записей
                msg += "*Первые 5 записей:*\n"
                for _, row in df.head(5).iterrows():
                    msg += f"  {int(row['year'])}: {row['value']:+.2f}\n"
                
                msg += "\n*Последние 5 записей:*\n"
                for _, row in df.tail(5).iterrows():
                    msg += f"  {int(row['year'])}: {row['value']:+.2f}\n"
                
                await update.message.reply_text(msg, parse_mode="Markdown")
            except Exception as e:
                await update.message.reply_text(f"✗ Ошибка загрузки {k}: {e}")
    except Exception as e:
        await update.message.reply_text(f"✗ Ошибка создания данных: {e}")

async def cmd_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /update"""
    indicators={'global_temp':(fetch_nasa_global_temp,'Temperature'), 'co2':(fetch_noaa_co2,'CO2')}
    results = []
    for k,(fn,_) in indicators.items():
        try:
            df=fn()
            save_indicator(df,k)
            results.append(f'✓ {k} обновлены')
        except Exception as e:
            results.append(f'✗ {k}: {str(e)[:100]}')
    await update.message.reply_text('\n'.join(results))

async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /analyze с выбором периода"""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    
    keyboard = [
        [InlineKeyboardButton("5 лет", callback_data="analyze_5")],
        [InlineKeyboardButton("10 лет", callback_data="analyze_10")],
        [InlineKeyboardButton("20 лет", callback_data="analyze_20")],
        [InlineKeyboardButton("50 лет", callback_data="analyze_50")],
        [InlineKeyboardButton("100 лет", callback_data="analyze_100")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    msg = "📊 Выберите период для анализа:"
    await update.message.reply_text(msg, reply_markup=reply_markup)

async def analyze_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка выбора периода анализа"""
    query = update.callback_query
    await query.answer()
    
    # Извлекаем количество лет из callback_data
    years = int(query.data.split('_')[1])
    
    indicators={'global_temp':(fetch_nasa_global_temp,'Temperature'), 'co2':(fetch_noaa_co2,'CO2')}
    try:
        loaded = {k: load_indicator(k) for k in indicators}
    except RuntimeError as e:
        await query.edit_message_text(f"✗ Ошибка: {e}")
        return
    
    summary={}
    for k,(_,name) in indicators.items():
        try:
            unit = UNITS.get(name, '')
            result=analyze_trend(loaded[k], years=years)
            if result:
                summary[name]=result
                msg = f"✓ *{name}* ({unit}):\n\n"
                msg += f"📊 *Анализ за {years} лет:*\n"
                msg += f"• Изменение: {result['abs_change']:+.3f} {unit.split()[0] if unit else ''}\n"
                msg += f"• Наклон: {result['slope_per_year']:+.4f}/год\n"
                msg += f"• R² (качество): {result['r2']:.2f}\n"
                if result['pct_change']:
                    msg += f"• Процент изм: {result['pct_change']:+.2f}%"
            else:
                msg = f"✗ {name}: недостаточно данных за {years} лет"
        except Exception as e:
            msg = f"✗ {name}: {e}"
        await context.bot.send_message(chat_id=query.message.chat_id, text=msg, parse_mode="Markdown")
    
    # Показываем легенду
    legend = "\n📋 *Расшифровка показателей:*\n"
    legend += "• *Изменение* — на сколько выросло/упало значение за период\n"
    legend += "• *Наклон* — как быстро меняется в год (единица/год)\n"
    legend += "• *R²* — тесноту связи (1.0 = идеально прямая, 0 = нет связи)\n"
    legend += "• *Процент изм* — процентное изменение от начального значения"
    await context.bot.send_message(chat_id=query.message.chat_id, text=legend, parse_mode="Markdown")

async def cmd_plot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /plot с выбором периода"""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    
    keyboard = [
        [InlineKeyboardButton("5 лет", callback_data="plot_5")],
        [InlineKeyboardButton("10 лет", callback_data="plot_10")],
        [InlineKeyboardButton("20 лет", callback_data="plot_20")],
        [InlineKeyboardButton("50 лет", callback_data="plot_50")],
        [InlineKeyboardButton("Все данные", callback_data="plot_all")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    msg = "📈 Выберите период для графика:"
    await update.message.reply_text(msg, reply_markup=reply_markup)

async def plot_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка выбора периода для графика"""
    query = update.callback_query
    await query.answer()
    
    # Извлекаем количество лет из callback_data
    data = query.data.split('_')[1]
    years = None if data == 'all' else int(data)
    
    indicators={'global_temp':(fetch_nasa_global_temp,'Temperature'), 'co2':(fetch_noaa_co2,'CO2')}
    try:
        loaded = {k: load_indicator(k) for k in indicators}
    except RuntimeError as e:
        await query.edit_message_text(f"✗ Ошибка: {e}")
        return
    
    await query.edit_message_text("⏳ Строю графики...")
    
    period_text = "все данные" if years is None else f"{years} лет"
    
    for k,(_,name) in indicators.items():
        try:
            plot_timeseries(loaded[k], name, name, years=years)
            out=OUT_DIR/f"{name.replace(' ','_')}.png"
            with open(out, 'rb') as photo:
                caption = f"📈 {name} ({period_text})"
                await context.bot.send_photo(chat_id=query.message.chat_id, photo=photo, caption=caption)
        except Exception as e:
            await context.bot.send_message(chat_id=query.message.chat_id, text=f"✗ Ошибка графика {name}: {e}")

async def cmd_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /report"""
    indicators={'global_temp':(fetch_nasa_global_temp,'Temperature'), 'co2':(fetch_noaa_co2,'CO2')}
    try:
        loaded = {k: load_indicator(k) for k in indicators}
    except RuntimeError as e:
        await update.message.reply_text(f"✗ Ошибка: {e}")
        return
    
    summary={}
    for k,(_,name) in indicators.items():
        try:
            summary[name]=analyze_trend(loaded[k],years=10)
        except:
            summary[name]=None
    
    report = make_report(summary)
    await update.message.reply_text(f"📊 *Полный отчет*\n\n{report}", parse_mode="Markdown")

async def cmd_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /all"""
    await update.message.reply_text("⏳ Запускаю полный цикл анализа...")
    
    try:
        create_sample_data()
        await update.message.reply_text("✓ Данные созданы")
    except Exception as e:
        await update.message.reply_text(f"✗ Ошибка создания данных: {e}")
        return
    
    indicators={'global_temp':(fetch_nasa_global_temp,'Temperature'), 'co2':(fetch_noaa_co2,'CO2')}
    try:
        loaded = {k: load_indicator(k) for k in indicators}
    except RuntimeError as e:
        await update.message.reply_text(f"✗ Ошибка: {e}")
        return
    
    # Анализ и графики
    summary={}
    for k,(_,name) in indicators.items():
        try:
            summary[name]=analyze_trend(loaded[k],years=10)
            plot_timeseries(loaded[k],name,name, years=10)
        except Exception as e:
            summary[name]=None
        
    # Графики
    for k,(_,name) in indicators.items():
        try:
            out=OUT_DIR/f"{name.replace(' ','_')}.png"
            with open(out, 'rb') as photo:
                await update.message.reply_photo(photo, caption=f"📈 {name} (10 лет)")
        except Exception as e:
            await update.message.reply_text(f"✗ Ошибка графика: {e}")
    
    # Отчет с легендой
    report = make_report(summary)
    await update.message.reply_text(f"📊 *Итоговый отчет*\n\n{report}", parse_mode="Markdown")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /help"""
    msg = """📖 *Справка*

*Climate Watcher* анализирует климатические данные:

🌡️ *Temperature* — глобальная температура в °C
💨 *CO2* — концентрация CO2 в ppm (части на миллион)

*Команды:*
• /sample — Создать пример данных (1980-2026)
• /update — Загрузить свежие данные из NASA/NOAA
• /analyze — Анализ тренда за 10 лет
• /plot — Построить графики
• /report — Показать отчет с единицами измерения
• /all — Полный цикл (создать + анализ + графики + отчет)
"""
    await update.message.reply_text(msg, parse_mode="Markdown")

def main_telegram():
    """Запуск Telegram бота"""
    TOKEN = os.getenv('TG_BOT_TOKEN')
    if not TOKEN:
        print("❌ Ошибка: переменная TG_BOT_TOKEN не найдена в .env файле")
        print("Создайте файл .env с содержимым:")
        print("  TG_BOT_TOKEN=ВАШ_ТОКЕН_ЗДЕСЬ")
        sys.exit(1)
    
    app = Application.builder().token(TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("sample", cmd_sample))
    app.add_handler(CommandHandler("update", cmd_update))
    app.add_handler(CommandHandler("analyze", cmd_analyze))
    app.add_handler(CommandHandler("plot", cmd_plot))
    app.add_handler(CommandHandler("report", cmd_report))
    app.add_handler(CommandHandler("all", cmd_all))
    app.add_handler(CommandHandler("help", cmd_help))
    
    # Обработчик для кнопок анализа
    from telegram.ext import CallbackQueryHandler
    app.add_handler(CallbackQueryHandler(analyze_callback, pattern=r'^analyze_\d+$'))
    app.add_handler(CallbackQueryHandler(plot_callback, pattern=r'^plot_(\d+|all)$'))
    
    print("🤖 Бот запущен. Нажмите Ctrl+C для остановки...")
    app.run_polling()

def main_cli():
    """Запуск в режиме CLI"""
    p=argparse.ArgumentParser(); p.add_argument('--init-sample',action='store_true'); p.add_argument('--update',action='store_true'); p.add_argument('--analyze',type=int,default=0); p.add_argument('--plot',action='store_true'); p.add_argument('--report',action='store_true'); args=p.parse_args()
    indicators={'global_temp':(fetch_nasa_global_temp,'Temperature'), 'co2':(fetch_noaa_co2,'CO2')}
    
    # Если нет аргументов, показать интерактивное меню
    if len(sys.argv)==1:
        while True:
            choice=interactive_menu()
            if choice=='0': print("До встречи!"); break
            elif choice=='1': create_sample_data()
            elif choice=='2':
                for k,(fn,_) in indicators.items():
                    try: df=fn(); save_indicator(df,k); print(f'✓ Обновлены данные {k}')
                    except Exception as e: print(f'✗ Ошибка {k}: {e}',file=sys.stderr)
            elif choice=='3':
                try: loaded={k:load_indicator(k) for k in indicators}
                except RuntimeError as e: print(f"Error: {e}", file=sys.stderr); continue
                years=input("За сколько лет анализировать? (по умолчанию 10): ").strip() or "10"
                try: years=int(years)
                except: years=10
                for k,(_,name) in indicators.items():
                    try: result=analyze_trend(loaded[k],years=years); print(f"✓ {name}: {result}")
                    except Exception as e: print(f"✗ {name}: {e}",file=sys.stderr)
            elif choice=='4':
                try: loaded={k:load_indicator(k) for k in indicators}
                except RuntimeError as e: print(f"Error: {e}", file=sys.stderr); continue
                for k,(_,name) in indicators.items(): plot_timeseries(loaded[k],name,name)
            elif choice=='5':
                try: loaded={k:load_indicator(k) for k in indicators}
                except RuntimeError as e: print(f"Error: {e}", file=sys.stderr); continue
                summary={}
                for k,(_,name) in indicators.items():
                    try: summary[name]=analyze_trend(loaded[k],years=10)
                    except: summary[name]=None
                print('\n--- Report ---\n'+make_report(summary))
            elif choice=='6':
                create_sample_data()
                try: loaded={k:load_indicator(k) for k in indicators}
                except RuntimeError as e: print(f"Error: {e}", file=sys.stderr); continue
                summary={}
                for k,(_,name) in indicators.items():
                    try: summary[name]=analyze_trend(loaded[k],years=10); plot_timeseries(loaded[k],name,name)
                    except Exception as e: print(f"✗ {name}: {e}",file=sys.stderr); summary[name]=None
                print('\n--- Report ---\n'+make_report(summary))
            else: print(" Неверный выбор. Попробуйте снова.")
        return
    
    if args.init_sample: create_sample_data()
    if args.update:
        for k,(fn,_) in indicators.items():
            try: df=fn(); save_indicator(df,k)
            except Exception as e: print(f'Failed {k}: {e}',file=sys.stderr)

    try:
        loaded = {k: load_indicator(k) for k in indicators}
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    summary = {}
    if args.analyze>0:
        for k,(_,name) in indicators.items():
            try:
                summary[name] = analyze_trend(loaded[k], years=args.analyze)
            except Exception as e:
                print(f"Analysis failed for {name}: {e}", file=sys.stderr)
                summary[name] = None
    if args.plot:
        for k,(_,name) in indicators.items():
            plot_timeseries(loaded[k], name, name)
    if args.report:
        print('\n--- Report ---\n', make_report(summary))

if __name__=='__main__':
    token = os.getenv('TG_BOT_TOKEN')
    try:
        if token:
            print("✓ Токен найден, запускаю Telegram бота...")
            main_telegram()
        else:
            print("⚠️  Токен не установлен (TG_BOT_TOKEN в .env)")
            print("\nСоздайте файл .env с содержимым:")
            print("  TG_BOT_TOKEN=ВАШ_ТОКЕН_ЗДЕСЬ")
            print("\nЗапускаю режим CLI (текстовое меню)...")
            print("-" * 50)
            main_cli()
    except KeyboardInterrupt:
        print("\n\n✓ Бот остановлен пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Ошибка: {e}")
        sys.exit(1)
