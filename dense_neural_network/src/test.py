from pre_process import PreProcess

def main():
    data = PreProcess.getDataArrayAsMap(dataFile='dummy.conll', onlyBioTagging=True)
    print(data)


if __name__ == "__main__":
    main()



