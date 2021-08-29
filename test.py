if __name__ == '__main__':
    writer = SummaryWriter('evaluation')
    load_path = './save_model'

    models = sorted([x.name for x in os.scandir(load_path) if x.name.startswith('DrebinGAN')])

    g_models = []
    d_models = []

    for model in models:
        if model.find('_D_') != -1:
            d_models.append(model)
        else:
            g_models.append(model)

    import re

    for i, _ in enumerate(g_models):
        with torch.no_grad():
            gan = DrebinGAN(classifier_model=DrebinSVM)
            gan.load(g_models[i], d_models[i])
            gan.evaluate(writer=writer, epoch=int(re.search('\d+', g_models[i]).group()))
        del gan